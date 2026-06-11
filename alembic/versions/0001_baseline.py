"""baseline schema — all 16 tables

Revision ID: 0001_baseline
Revises: None
Create Date: 2026-06-08 00:00:00.000000

Single baseline migration consolidating all prior migrations.
Creates all 16 tables, indexes, constraints, and TimescaleDB hypertable.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001_baseline"
down_revision = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    # ─── Create ENUM types ────────────────────────────────────────────
    op.execute("CREATE TYPE alarm_status_enum AS ENUM ('NORMAL', 'UNACKNOWLEDGED', 'ACKNOWLEDGED', 'NORMAL_UNACK', 'SHELVED', 'OUT_OF_SERVICE', 'SUPPRESSED')")
    op.execute("CREATE TYPE severity_enum AS ENUM ('WARNING', 'CRITICAL')")
    op.execute("CREATE TYPE decision_action_enum AS ENUM ('APPROVE', 'DEFER', 'REJECT')")
    op.execute("CREATE TYPE work_order_priority_enum AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')")
    op.execute("CREATE TYPE work_order_status_enum AS ENUM ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED')")

    # ─── Table 1: sensor_readings ─────────────────────────────────────
    op.create_table(
        "sensor_readings",
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("sensor_name", sa.String(50), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("upstream_effect", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("machine_phase", sa.String(20), nullable=True),
        sa.Column("fft_data", sa.JSON(), nullable=True),
        sa.Column("present", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.PrimaryKeyConstraint("machine_id", "sensor_name", "timestamp"),
    )
    op.create_index("ix_sensor_readings_machine_time", "sensor_readings", ["machine_id", "timestamp"])
    op.create_index(
        "ix_sensor_readings_anomaly",
        "sensor_readings",
        ["machine_id", "is_anomaly"],
        postgresql_where=sa.text("is_anomaly = true"),
    )

    op.execute("""
        SELECT create_hypertable(
            'sensor_readings',
            'timestamp',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)

    # ─── Table 2: machine_baselines ───────────────────────────────────
    op.create_table(
        "machine_baselines",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("sensor", sa.String(50), nullable=False),
        sa.Column("mean_value", sa.Float(), nullable=False),
        sa.Column("std_value", sa.Float(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("calibrated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.UniqueConstraint("machine_id", "sensor", "is_active", name="uq_active_baseline"),
    )
    op.create_index("ix_machine_baselines_machine_id", "machine_baselines", ["machine_id"])

    # ─── Table 3: anomaly_log ─────────────────────────────────────────
    op.create_table(
        "anomaly_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("detected_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("shap_values", sa.JSON(), nullable=True),
        sa.Column("top_contributing_sensor", sa.String(50), nullable=True),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'ACTIVE'")),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("resolution_type", sa.String(30), nullable=True),
        sa.Column("upstream_effect", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("fault_type", sa.String(50), nullable=True),
        sa.Column("fault_confidence", sa.Float(), nullable=True),
        sa.CheckConstraint("severity IN ('WARNING', 'CRITICAL')", name="ck_anomaly_log_severity"),
    )
    op.create_index("ix_anomaly_log_machine_id", "anomaly_log", ["machine_id"])
    op.create_index("ix_anomaly_log_detected_at", "anomaly_log", ["detected_at"])

    # ─── Table 4: alarm_state ────────────────────────────────────────
    op.create_table(
        "alarm_state",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("anomaly_id", sa.BigInteger(), sa.ForeignKey("anomaly_log.id"), nullable=False),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(25), nullable=False, server_default=sa.text("'UNACKNOWLEDGED'")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_updated", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("trend_direction", sa.String(15), nullable=True),
        sa.Column("trend_check_count", sa.Integer(), server_default=sa.text("0")),
        sa.Column("escalated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("shelved_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("shelved_until", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("shelve_reason", sa.Text(), nullable=True),
        sa.Column("shelved_by_role", sa.String(30), nullable=True),
        sa.Column("oos_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("work_order_id", sa.String(50), nullable=True),
        sa.Column("oos_restored_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("suppressed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("suppressed_by", sa.String(30), nullable=True),
        sa.Column("suppress_reason", sa.Text(), nullable=True),
        sa.Column("suppress_review_due", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('NORMAL', 'UNACKNOWLEDGED', 'ACKNOWLEDGED', 'NORMAL_UNACK', 'SHELVED', 'OUT_OF_SERVICE', 'SUPPRESSED')",
            name="ck_alarm_state_status",
        ),
    )
    op.create_index("ix_alarm_state_anomaly_id", "alarm_state", ["anomaly_id"])
    op.create_index("ix_alarm_state_machine_id", "alarm_state", ["machine_id"])
    op.create_index(
        "ix_alarm_state_active",
        "alarm_state",
        ["machine_id", "created_at"],
        postgresql_where=sa.text("status NOT IN ('NORMAL', 'SUPPRESSED', 'OUT_OF_SERVICE')"),
    )

    # ─── Table 5: alarm_state_transitions ───────────────────────────
    op.create_table(
        "alarm_state_transitions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("alarm_id", sa.BigInteger(), sa.ForeignKey("alarm_state.id"), nullable=False),
        sa.Column("from_state", sa.String(25), nullable=False),
        sa.Column("to_state", sa.String(25), nullable=False),
        sa.Column("operator_role", sa.String(30), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_alarm_state_transitions_alarm_id", "alarm_state_transitions", ["alarm_id"])
    op.create_index("ix_alarm_transitions_timestamp", "alarm_state_transitions", ["timestamp"])

    # ─── Table 6: decision_log ──────────────────────────────────────
    op.create_table(
        "decision_log",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("alarm_id", sa.BigInteger(), sa.ForeignKey("alarm_state.id"), nullable=True),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("action", sa.String(50), nullable=False, server_default=sa.text("'PENDING'")),
        sa.Column("scenario_id", sa.String(10), nullable=True),
        sa.Column("chosen_scenario_id", sa.String(10), nullable=True),
        sa.Column("operator_role", sa.String(30), nullable=True),
        sa.Column("decided_by", sa.String(30), nullable=True),
        sa.Column("decided_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("response_time_s", sa.Integer(), nullable=True),
        sa.Column("ai_recommendation", sa.String(10), nullable=True),
        sa.Column("overridden", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("escalation_level", sa.Integer(), server_default=sa.text("1")),
        sa.Column("auto_approved", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("resolution_source", sa.String(30), nullable=True),
        sa.Column("outcome", sa.String(30), nullable=True),
        sa.Column("due_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("scenarios_presented", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("downtime_minutes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_decision_log_alarm_id", "decision_log", ["alarm_id"])
    op.create_index("ix_audit_machine_time", "decision_log", ["alarm_id", "created_at"])
    op.create_index(
        "ix_decision_log_pending",
        "decision_log",
        ["machine_id", "created_at"],
        postgresql_where=sa.text("action = 'PENDING'"),
    )

    # ─── Table 6b: decision_audit_log ───────────────────────────────
    op.create_table(
        "decision_audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("decision_id", sa.String(36), sa.ForeignKey("decision_log.id"), nullable=False),
        sa.Column("alarm_id", sa.BigInteger(), sa.ForeignKey("alarm_state.id"), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("scenario_id", sa.String(10), nullable=True),
        sa.Column("operator_role", sa.String(30), nullable=True),
        sa.Column("response_time_s", sa.Integer(), nullable=True),
        sa.Column("ai_recommendation", sa.String(10), nullable=True),
        sa.Column("overridden", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("override_reason", sa.Text(), nullable=True),
        sa.Column("escalation_level", sa.Integer(), server_default=sa.text("1")),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_decision_audit_log_decision_id", "decision_audit_log", ["decision_id"])
    op.create_index("ix_decision_audit_log_alarm_id", "decision_audit_log", ["alarm_id"])
    op.create_index("ix_decision_audit_decision_id", "decision_audit_log", ["decision_id"])
    op.create_index("ix_decision_audit_created_at", "decision_audit_log", ["created_at"])

    # ─── Table 7: machine_health_score ──────────────────────────────
    op.create_table(
        "machine_health_score",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("calculated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("health_score", sa.Float(), nullable=False),
        sa.Column("availability_score", sa.Float(), nullable=True),
        sa.Column("reliability_score", sa.Float(), nullable=True),
        sa.Column("condition_score", sa.Float(), nullable=True),
        sa.Column("rul_hours", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("classification", sa.String(30), nullable=True),
        sa.CheckConstraint("health_score >= 0.0 AND health_score <= 1.0", name="ck_machine_health_score_range"),
    )
    op.create_index("ix_machine_health_score_calculated_at", "machine_health_score", ["calculated_at"])
    op.create_index("ix_mhi_machine_time", "machine_health_score", ["machine_id", "calculated_at"])

    # ─── Table 8: canary_probe_log ──────────────────────────────────
    op.create_table(
        "canary_probe_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("probe_id", sa.String(36), nullable=False, unique=True),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("probe_type", sa.String(10), nullable=False),
        sa.Column("scenario", sa.String(100), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("detected", sa.Boolean(), nullable=True),
        sa.Column("expected", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("duration_s", sa.Integer(), nullable=True),
        sa.Column("triggered_by", sa.String(30), nullable=False),
        sa.Column("recalibration_triggered", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_canary_machine_time", "canary_probe_log", ["machine_id", "started_at"])

    # ─── Table 9: settings ──────────────────────────────────────────
    op.create_table(
        "settings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("category", sa.String(20), nullable=False),
        sa.Column("key", sa.String(60), nullable=False),
        sa.Column("value", sa.String(255), nullable=False),
        sa.Column("unit", sa.String(20), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_by", sa.String(30), nullable=True),
        sa.UniqueConstraint("category", "key", name="uq_settings_category_key"),
    )

    # ─── Table 10: maintenance_log ──────────────────────────────────
    op.create_table(
        "maintenance_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("alarm_id", sa.BigInteger(), sa.ForeignKey("alarm_state.id"), nullable=True),
        sa.Column("scheduled_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("performed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("technician_notes", sa.Text(), nullable=True),
        sa.Column("fault_found", sa.Boolean(), nullable=True),
        sa.Column("fault_description", sa.Text(), nullable=True),
        sa.Column("downtime_minutes", sa.Integer(), nullable=True),
        sa.Column("cost_eur", sa.Float(), nullable=True),
    )
    op.create_index("ix_maintenance_log_alarm_id", "maintenance_log", ["alarm_id"])
    op.create_index("ix_maintenance_machine_time", "maintenance_log", ["machine_id", "performed_at"])

    # ─── Table 11: work_orders ──────────────────────────────────────
    op.create_table(
        "work_orders",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("decision_id", sa.String(36), sa.ForeignKey("decision_log.id"), nullable=True),
        sa.Column("alarm_id", sa.BigInteger(), sa.ForeignKey("alarm_state.id"), nullable=True),
        sa.Column("fault_type", sa.String(50), nullable=True),
        sa.Column("recommended_action", sa.Text(), nullable=True),
        sa.Column("action_type", sa.String(50), nullable=True),
        sa.Column("priority", sa.String(10), nullable=False, server_default=sa.text("'MEDIUM'")),
        sa.Column("estimated_cost_eur", sa.Float(), nullable=True),
        sa.Column("assigned_team", sa.String(50), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'OPEN'")),
        sa.Column("created_by", sa.String(50), nullable=True),
        sa.Column("scenario_id", sa.String(10), nullable=True),
        sa.Column("work_order_number", sa.String(30), nullable=True, unique=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_work_orders_machine_id", "work_orders", ["machine_id"])
    op.create_index("ix_work_orders_decision_id", "work_orders", ["decision_id"])
    op.create_index("ix_work_orders_alarm_id", "work_orders", ["alarm_id"])
    op.create_index("ix_work_orders_status", "work_orders", ["status"])
    op.create_index("ix_work_orders_created_at", "work_orders", ["created_at"])

    # ─── Table 12: shift_reports ────────────────────────────────────
    op.create_table(
        "shift_reports",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("shift_start", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("shift_end", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("shift_type", sa.String(10), nullable=False),
        sa.Column("generated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("report_data", sa.JSON(), nullable=False),
        sa.Column("html_content", sa.Text(), nullable=True),
    )
    op.create_index("ix_shift_reports_start", "shift_reports", ["shift_start"])

    # ─── Table 13: ai_act_log ───────────────────────────────────────
    op.create_table(
        "ai_act_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("model_version", sa.String(30), nullable=False),
        sa.Column("decision_type", sa.String(50), nullable=False),
        sa.Column("features_snapshot", sa.JSON(), nullable=True),
        sa.Column("output", sa.JSON(), nullable=True),
        sa.Column("shap_values", sa.JSON(), nullable=True),
        sa.Column("action_taken", sa.String(100), nullable=True),
        sa.Column("human_action", sa.String(100), nullable=True),
    )
    op.create_index("ix_ai_act_log_timestamp", "ai_act_log", ["timestamp"])
    op.create_index("ix_ai_act_log_machine_time", "ai_act_log", ["machine_id", "timestamp"])

    # ─── Table 14: decision_audit (Phase 7 orphan) ──────────────────
    op.create_table(
        "decision_audit",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("alarm_id", sa.String(50), nullable=False),
        sa.Column("state_from", sa.String(30), nullable=False),
        sa.Column("state_to", sa.String(30), nullable=False),
        sa.Column("ml_inputs", JSONB(), nullable=True),
        sa.Column("diagnosis", JSONB(), nullable=True),
        sa.Column("options_offered", JSONB(), nullable=True),
        sa.Column("recommended_option_id", sa.String(30), nullable=True),
        sa.Column("operator_choice", sa.String(30), nullable=True),
        sa.Column("operator_id", sa.String(50), nullable=True),
        sa.Column("operator_type", sa.String(10), nullable=False),
        sa.Column("decision_source", sa.String(20), nullable=False),
        sa.Column("operator_reason", sa.Text(), nullable=True),
        sa.Column("severity", sa.String(10), nullable=True),
        sa.Column("time_to_action_minutes", sa.Integer(), nullable=True),
    )
    op.create_index(
        "idx_decision_audit_machine_time",
        "decision_audit",
        ["machine_id", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_decision_audit_alarm_time",
        "decision_audit",
        ["alarm_id", sa.text("timestamp DESC")],
    )

    # ─── Table 15: decision_work_orders (Phase 7 orphan) ────────────
    op.create_table(
        "decision_work_orders",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("alarm_id", sa.String(50), nullable=True),
        sa.Column("decision_audit_id", sa.BigInteger(), sa.ForeignKey("decision_audit.id", ondelete="SET NULL"), nullable=True),
        sa.Column("work_order_type", sa.String(30), nullable=False),
        sa.Column("priority", sa.String(10), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("assigned_to", sa.String(100), nullable=True),
        sa.Column("created_by", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'open'")),
        sa.Column("estimated_cost_saved", sa.Numeric(12, 2), nullable=True),
        sa.Column("closed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("close_notes", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_decision_work_orders_machine_time",
        "decision_work_orders",
        ["machine_id", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_decision_work_orders_status_priority",
        "decision_work_orders",
        ["status", "priority"],
    )

    # ─── COMMENT ON documentation ────────────────────────────────────
    op.execute("COMMENT ON TABLE sensor_readings IS 'TimescaleDB hypertable for raw sensor readings at 10s intervals'")
    op.execute("COMMENT ON COLUMN sensor_readings.value IS 'Sensor reading value in native units'")
    op.execute("COMMENT ON COLUMN sensor_readings.is_anomaly IS 'Flag indicating anomalous reading detected by ML model'")
    op.execute("COMMENT ON TABLE machine_baselines IS 'Calibrated baseline statistics per machine-sensor pair'")
    op.execute("COMMENT ON TABLE anomaly_log IS 'Recorded anomaly events from ML detection pipeline'")
    op.execute("COMMENT ON COLUMN anomaly_log.severity IS 'Anomaly severity classified by ML model'")
    op.execute("COMMENT ON TABLE alarm_state IS 'Current alarm state machine tracking anomaly lifecycle'")
    op.execute("COMMENT ON COLUMN alarm_state.status IS 'Alarm state machine position'")
    op.execute("COMMENT ON TABLE alarm_state_transitions IS 'Audit trail of all alarm state transitions'")
    op.execute("COMMENT ON TABLE decision_log IS 'Decision records for operator or AI-driven actions on alarms'")
    op.execute("COMMENT ON COLUMN decision_log.action IS 'Decision action taken or pending'")
    op.execute("COMMENT ON TABLE decision_audit_log IS 'Audit trail of decision mutations and overrides'")
    op.execute("COMMENT ON TABLE machine_health_score IS 'Periodic machine health scores with sub-scores and RUL'")
    op.execute("COMMENT ON TABLE canary_probe_log IS 'Canary probe results for ML pipeline health monitoring'")
    op.execute("COMMENT ON TABLE settings IS 'Runtime configuration key-value store'")
    op.execute("COMMENT ON TABLE maintenance_log IS 'Maintenance activity records linked to alarms'")
    op.execute("COMMENT ON TABLE work_orders IS 'Work orders generated from decision outcomes'")
    op.execute("COMMENT ON COLUMN work_orders.priority IS 'Work order priority level'")
    op.execute("COMMENT ON COLUMN work_orders.status IS 'Work order lifecycle status'")
    op.execute("COMMENT ON TABLE shift_reports IS 'Generated shift summary reports'")
    op.execute("COMMENT ON TABLE ai_act_log IS 'EU AI Act compliance log for automated decisions'")
    op.execute("COMMENT ON TABLE decision_audit IS 'Phase 7 decision audit trail with full ML context'")
    op.execute("COMMENT ON TABLE decision_work_orders IS 'Phase 7 work orders linked to decision audit'")


def downgrade() -> None:
    op.drop_table("decision_work_orders")
    op.drop_table("decision_audit")
    op.drop_table("ai_act_log")
    op.drop_table("shift_reports")
    op.drop_table("work_orders")
    op.drop_table("maintenance_log")
    op.drop_table("settings")
    op.drop_table("canary_probe_log")
    op.drop_table("machine_health_score")
    op.drop_table("decision_audit_log")
    op.drop_table("decision_log")
    op.drop_table("alarm_state_transitions")
    op.drop_table("alarm_state")
    op.drop_table("anomaly_log")
    op.drop_table("machine_baselines")
    op.drop_table("sensor_readings")
    op.execute("DROP TYPE IF EXISTS work_order_status_enum")
    op.execute("DROP TYPE IF EXISTS work_order_priority_enum")
    op.execute("DROP TYPE IF EXISTS decision_action_enum")
    op.execute("DROP TYPE IF EXISTS severity_enum")
    op.execute("DROP TYPE IF EXISTS alarm_status_enum")
