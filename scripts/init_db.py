"""
scripts/init_db.py
==================
Idempotent database initializer. Safe to run multiple times.

Usage:
    python scripts/init_db.py

Steps:
   1. run_migrations()                  — fail closed unless the schema reaches Alembic HEAD
   2. create_all()                      — create any idempotent model-owned tables
   3. create_hypertable()               — TimescaleDB hypertable on sensor_readings
   4. configure_compression()           — auto-compress sensor_readings older than 7 days
   5. configure_mhi_compression()       — auto-compress machine_health_score older than 30 days
   6. configure_ai_act_compression()    — auto-compress ai_act_log older than 14 days
   7. configure_anomaly_compression()   — auto-compress anomaly_log older than 14 days
   8. configure_retention()             — auto-drop chunks per table retention policy
   9. create_continuous_aggregates()    — 4 materialized views for time-series rollups
  10. configure_refresh_policies()      — auto-refresh schedules for all CAGGs
  11. seed_default_settings()           — 13 financial + physics parameters
  12. seed_health_scores()              — 6 machines with MHI=1.0 (Excellent)
  13. verify()                          — final sanity check
"""

import os
import subprocess
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from src.database.connection import get_db_context, get_engine
from src.database.models import Base, MachineHealthScore, Settings

engine = get_engine()


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


configure_console_output()


MACHINE_IDS = ["AC-201", "HX-202", "CM-203", "AC-301", "HX-302", "CM-303"]


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables created (16 total)")


def create_timescale_hypertable():
    with engine.connect() as conn:
        conn.execute(text("""
            SELECT create_hypertable(
                'sensor_readings',
                'timestamp',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE
            );
        """))
        conn.commit()
    print("TimescaleDB hypertable created: sensor_readings")


def configure_compression():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM timescaledb_information.compression_settings
                WHERE hypertable_name = 'sensor_readings'
            )
        """))
        already_compressed = result.fetchone()[0]
        if already_compressed:
            print('  Compression already configured, skipping...')
            return

        conn.execute(text("""
            ALTER TABLE sensor_readings
            SET (
                timescaledb.compress,
                timescaledb.compress_orderby   = 'timestamp DESC',
                timescaledb.compress_segmentby = 'machine_id'
            );
        """))
        conn.execute(text("""
            SELECT add_compression_policy(
                'sensor_readings',
                INTERVAL '7 days',
                if_not_exists => TRUE
            );
        """))
        conn.commit()
    print("Compression policy configured: compress after 7 days")


def _ensure_hypertable(table: str, time_column: str):
    """Idempotently convert a table with an integer PK into a hypertable.

    TimescaleDB requires every unique constraint to include the partition
    column, so the PK is widened to (id, time_column) on first conversion.
    Tables referenced by foreign keys (e.g. anomaly_log) cannot be converted.
    """
    with engine.connect() as conn:
        already = conn.execute(
            text("SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = :t"),
            {"t": table},
        ).first()
        if already:
            return
        # Identifiers cannot be bound parameters; table/time_column are
        # internal constants, not user input.
        conn.execute(text(
            "ALTER TABLE {t} DROP CONSTRAINT IF EXISTS {t}_pkey".format(t=table)
        ))
        conn.execute(text(
            "ALTER TABLE {t} ADD PRIMARY KEY (id, {c})".format(t=table, c=time_column)
        ))
        conn.execute(
            text(
                "SELECT create_hypertable(:t, :col, "
                "chunk_time_interval => INTERVAL '1 day', "
                "migrate_data => TRUE, if_not_exists => TRUE)"
            ),
            {"t": table, "col": time_column},
        )
        conn.commit()
    print("Hypertable created: {} on {}".format(table, time_column))


def configure_mhi_compression():
    _ensure_hypertable("machine_health_score", "calculated_at")
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE machine_health_score
            SET (
                timescaledb.compress,
                timescaledb.compress_orderby   = 'calculated_at DESC',
                timescaledb.compress_segmentby = 'machine_id'
            );
        """))
        conn.execute(text("""
            SELECT add_compression_policy(
                'machine_health_score',
                INTERVAL '30 days',
                if_not_exists => TRUE
            );
        """))
        conn.commit()
    print("Compression policy configured: machine_health_score compress after 30 days")


def configure_ai_act_compression():
    _ensure_hypertable("ai_act_log", "timestamp")
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE ai_act_log
            SET (
                timescaledb.compress,
                timescaledb.compress_orderby   = 'timestamp DESC',
                timescaledb.compress_segmentby = 'machine_id'
            );
        """))
        conn.execute(text("""
            SELECT add_compression_policy(
                'ai_act_log',
                INTERVAL '14 days',
                if_not_exists => TRUE
            );
        """))
        conn.commit()
    print("Compression policy configured: ai_act_log compress after 14 days")


def configure_anomaly_compression():
    # anomaly_log is referenced by alarm_state.anomaly_id (FK), and TimescaleDB
    # does not support foreign keys pointing at hypertables — so it stays a
    # regular table. Volume is low (one row per detected anomaly), compression
    # is unnecessary.
    print("Skipped: anomaly_log stays a regular table (FK target; low volume)")


def configure_retention():
    retention_days = int(os.getenv("DATA_RETENTION_DAYS", "90"))
    with engine.connect() as conn:
        conn.execute(
            text("SELECT add_retention_policy('sensor_readings', INTERVAL :days, if_not_exists => TRUE);"),
            {"days": "{} days".format(retention_days)},
        )
        conn.execute(text("""
            SELECT add_retention_policy(
                'machine_health_score',
                INTERVAL '365 days',
                if_not_exists => TRUE
            );
        """))
        conn.execute(text("""
            SELECT add_retention_policy(
                'ai_act_log',
                INTERVAL '365 days',
                if_not_exists => TRUE
            );
        """))
        # anomaly_log is not a hypertable (FK target) — no retention policy.
        conn.commit()
    print("Retention policy configured: drop after {} days".format(retention_days))


def create_continuous_aggregates():
    # CREATE MATERIALIZED VIEW ... WITH DATA must run outside a transaction.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_sensor_15min
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('15 minutes', timestamp) AS bucket,
                machine_id,
                sensor_name,
                AVG(value) AS avg_value,
                MIN(value) AS min_value,
                MAX(value) AS max_value,
                STDDEV(value) AS stddev_value,
                COUNT(*) AS sample_count
            FROM sensor_readings
            GROUP BY bucket, machine_id, sensor_name;
        """))
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_sensor_1hour
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', bucket) AS bucket,
                machine_id,
                sensor_name,
                AVG(avg_value) AS avg_value,
                MIN(min_value) AS min_value,
                MAX(max_value) AS max_value,
                SUM(sample_count) AS sample_count
            FROM cagg_sensor_15min
            GROUP BY 1, 2, 3;
        """))
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_mhi_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', calculated_at) AS bucket,
                machine_id,
                AVG(health_score) AS avg_health_score,
                AVG(availability_score) AS avg_availability,
                AVG(reliability_score) AS avg_reliability,
                AVG(condition_score) AS avg_condition
            FROM machine_health_score
            GROUP BY bucket, machine_id;
        """))
        # anomaly_log is not a hypertable, so this is a plain materialized
        # view (refreshed on demand) rather than a continuous aggregate.
        conn.execute(text("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_anomaly_rate_hourly AS
            SELECT
                time_bucket('1 hour', detected_at) AS bucket,
                machine_id,
                COUNT(*) AS anomaly_count,
                AVG(anomaly_score) AS avg_anomaly_score
            FROM anomaly_log
            GROUP BY bucket, machine_id;
        """))
    print("Continuous aggregates created: cagg_sensor_15min, cagg_sensor_1hour, cagg_mhi_hourly, cagg_anomaly_rate_hourly")


def configure_refresh_policies():
    with engine.connect() as conn:
        conn.execute(text("""
            SELECT add_continuous_aggregate_policy('cagg_sensor_15min',
                start_offset => INTERVAL '1 hour',
                end_offset => INTERVAL '1 minute',
                schedule_interval => INTERVAL '15 minutes',
                if_not_exists => TRUE);
        """))
        conn.execute(text("""
            SELECT add_continuous_aggregate_policy('cagg_sensor_1hour',
                start_offset => INTERVAL '3 hours',
                end_offset => INTERVAL '1 minute',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE);
        """))
        conn.execute(text("""
            SELECT add_continuous_aggregate_policy('cagg_mhi_hourly',
                start_offset => INTERVAL '3 hours',
                end_offset => INTERVAL '1 minute',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE);
        """))
        # cagg_anomaly_rate_hourly is a plain materialized view — no
        # continuous-aggregate refresh policy applies.
        conn.commit()
    print("Refresh policies configured for all continuous aggregates")


def seed_default_settings():
    defaults = [
        ("financial", "line_a_hourly_production_eur",  "850",   "EUR/h",
         "Hourly production value for Line A"),
        ("financial", "line_b_hourly_production_eur",  "720",   "EUR/h",
         "Hourly production value for Line B"),
        ("financial", "ac_emergency_repair_eur",       "2850",  "EUR",
         "Emergency repair cost for air compressor"),
        ("financial", "hx_emergency_repair_eur",       "1800",  "EUR",
         "Emergency repair cost for heat exchanger"),
        ("financial", "cm_emergency_repair_eur",       "1400",  "EUR",
         "Emergency repair cost for conveyor motor"),
        ("financial", "planned_vs_emergency_ratio",    "0.35",  "x",
         "Planned maintenance cost as ratio of emergency cost"),
        ("financial", "emergency_premium_mult",        "1.8",   "x",
         "Cost multiplier for unplanned emergency work"),
        ("physics",  "degradation_rate_sigma",         "0.15",  "-",
         "Sigma for degradation rate Gaussian noise"),
        ("physics",  "sensor_noise_sigma",             "0.05",  "-",
         "Sigma for sensor reading noise"),
        ("physics",  "weibull_variation",              "0.10",  "-",
         "Weibull parameter variation coefficient"),
        ("physics",  "canary_probe_frequency_per_week","2",     "count",
         "Number of canary probes per week (Tue + Fri)"),
        ("physics",  "canary_probe_hour_utc",          "2",     "hour",
         "UTC hour to run scheduled canary probes"),
        ("physics",  "auto_recalibrate",               "true",  "bool",
         "Automatically recalibrate ML models on drift detection"),
    ]

    with get_db_context() as db:
        added = 0
        for category, key, value, unit, description in defaults:
            existing = db.query(Settings).filter_by(
                category=category, key=key
            ).first()
            if not existing:
                db.add(Settings(
                    category=category,
                    key=key,
                    value=value,
                    unit=unit,
                    description=description,
                    updated_by="system",
                ))
                added += 1
        db.commit()
    print("Settings seeded: {} new, {} already existed".format(added, len(defaults) - added))


def seed_initial_health_scores():
    now = datetime.now(UTC).replace(tzinfo=None)
    with get_db_context() as db:
        for machine_id in MACHINE_IDS:
            existing = db.query(MachineHealthScore).filter_by(
                machine_id=machine_id
            ).first()
            if not existing:
                db.add(MachineHealthScore(
                    machine_id=machine_id,
                    calculated_at=now,
                    health_score=1.0,
                    availability_score=1.0,
                    reliability_score=1.0,
                    condition_score=1.0,
                    rul_hours=None,
                    confidence=None,
                    classification="Excellent",
                ))
        db.commit()
    print("Initial MHI scores seeded (6 machines, Excellent)")


def verify():
    with get_db_context() as db:
        result = db.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
        """)).scalar()
        print("  Tables in public schema: {}".format(result))

        result = db.execute(text("""
            SELECT COUNT(*) FROM timescaledb_information.hypertables
            WHERE hypertable_name = 'sensor_readings'
        """)).scalar()
        assert result == 1, "sensor_readings hypertable not found!"
        print("  sensor_readings is a hypertable")

        count = db.query(Settings).count()
        assert count >= 13, "Expected 13 settings, found {}".format(count)
        print("  Settings: {} rows".format(count))

        count = db.query(MachineHealthScore).count()
        assert count >= 6, "Expected at least 6 MHI rows, found {}".format(count)
        print("  MachineHealthScore: {} rows".format(count))

    print("\nDatabase initialized — system ready")


def run_migrations():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "no migration error details reported"
            raise RuntimeError("Alembic migration failed: {}".format(detail))
        print("Alembic migrations applied")
    except OSError as exc:
        raise RuntimeError("Unable to start Alembic migrations: {}".format(exc)) from exc


if __name__ == "__main__":
    print("=== PdM Intelligence v3 — DB Init ===\n")
    run_migrations()
    create_tables()
    create_timescale_hypertable()
    configure_compression()
    configure_mhi_compression()
    configure_ai_act_compression()
    configure_anomaly_compression()
    configure_retention()
    create_continuous_aggregates()
    configure_refresh_policies()
    seed_default_settings()
    seed_initial_health_scores()
    print("\n--- Verification ---")
    verify()
