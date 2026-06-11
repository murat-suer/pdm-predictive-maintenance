import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

INIT_DB_PATH = Path(__file__).resolve().parents[3] / "scripts" / "init_db.py"
MODELS_PATH = Path(__file__).resolve().parents[3] / "src" / "database" / "models.py"


def _read_init_db():
    return INIT_DB_PATH.read_text(encoding="utf-8")


def _read_models():
    return MODELS_PATH.read_text(encoding="utf-8")


class TestCompressionPolicies:
    def test_sensor_readings_compress_after_7_days(self):
        content = _read_init_db()
        assert "add_compression_policy" in content
        assert "sensor_readings" in content
        assert "7 days" in content

    def test_mhi_compress_after_30_days(self):
        content = _read_init_db()
        assert "configure_mhi_compression" in content
        assert "machine_health_score" in content
        assert "30 days" in content

    def test_ai_act_compress_after_14_days(self):
        content = _read_init_db()
        assert "configure_ai_act_compression" in content
        assert "ai_act_log" in content
        assert "14 days" in content

    def test_anomaly_log_compress_after_14_days(self):
        content = _read_init_db()
        assert "configure_anomaly_compression" in content
        assert "anomaly_log" in content


class TestRetentionPolicies:
    def test_sensor_readings_retain_90_days(self):
        content = _read_init_db()
        assert "add_retention_policy" in content
        assert "90" in content

    def test_mhi_retain_365_days(self):
        content = _read_init_db()
        assert "365 days" in content

    def test_ai_act_retain_365_days(self):
        content = _read_init_db()
        assert "configure_ai_act_compression" in content

    def test_anomaly_log_retain_365_days(self):
        content = _read_init_db()
        assert "configure_anomaly_compression" in content


class TestContinuousAggregates:
    def test_cagg_sensor_15min_exists(self):
        content = _read_init_db()
        assert "cagg_sensor_15min" in content
        assert "time_bucket" in content
        assert "15 minutes" in content

    def test_cagg_sensor_1hour_exists(self):
        content = _read_init_db()
        assert "cagg_sensor_1hour" in content
        assert "1 hour" in content

    def test_cagg_mhi_hourly_exists(self):
        content = _read_init_db()
        assert "cagg_mhi_hourly" in content
        assert "health_score" in content

    def test_cagg_anomaly_rate_hourly_exists(self):
        content = _read_init_db()
        assert "cagg_anomaly_rate_hourly" in content
        assert "anomaly_count" in content

    def test_caggs_use_timescaledb_continuous(self):
        content = _read_init_db()
        assert "timescaledb.continuous" in content

    def test_create_continuous_aggregates_function_exists(self):
        content = _read_init_db()
        assert "create_continuous_aggregates" in content


class TestRefreshPolicies:
    def test_cagg_15min_refresh_policy(self):
        content = _read_init_db()
        assert "add_continuous_aggregate_policy" in content
        assert "cagg_sensor_15min" in content

    def test_cagg_1hour_refresh_policy(self):
        content = _read_init_db()
        assert "cagg_sensor_1hour" in content

    def test_cagg_mhi_refresh_policy(self):
        content = _read_init_db()
        assert "cagg_mhi_hourly" in content

    def test_cagg_anomaly_refresh_policy(self):
        content = _read_init_db()
        assert "cagg_anomaly_rate_hourly" in content

    def test_refresh_schedule_intervals(self):
        content = _read_init_db()
        assert "schedule_interval" in content
        assert "start_offset" in content
        assert "end_offset" in content


class TestCaggModels:
    def test_sensor_aggregate_15min_model_exists(self):
        from src.database.models import SensorAggregate15Min
        assert SensorAggregate15Min.__tablename__ == "cagg_sensor_15min"

    def test_sensor_aggregate_1hour_model_exists(self):
        from src.database.models import SensorAggregate1Hour
        assert SensorAggregate1Hour.__tablename__ == "cagg_sensor_1hour"

    def test_mhi_hourly_model_exists(self):
        from src.database.models import MHIHourly
        assert MHIHourly.__tablename__ == "cagg_mhi_hourly"

    def test_anomaly_rate_hourly_model_exists(self):
        from src.database.models import AnomalyRateHourly
        assert AnomalyRateHourly.__tablename__ == "cagg_anomaly_rate_hourly"

    def test_cagg_models_are_read_only(self):
        from src.database.models import SensorAggregate15Min
        info = SensorAggregate15Min.__table_args__
        if isinstance(info, dict):
            assert info.get("info", {}).get("materialized_view") is True
        else:
            found = False
            for arg in info:
                if isinstance(arg, dict) and arg.get("info", {}).get("materialized_view") is True:
                    found = True
            assert found, "CAGG models must be marked as materialized_view"

    def test_sensor_aggregate_15min_has_required_columns(self):
        from src.database.models import SensorAggregate15Min
        col_names = {c.name for c in SensorAggregate15Min.__table__.columns}
        required = {"bucket", "machine_id", "sensor_name", "avg_value", "min_value", "max_value", "stddev_value", "sample_count"}
        missing = required - col_names
        assert not missing, f"SensorAggregate15Min missing columns: {missing}"

    def test_sensor_aggregate_1hour_has_required_columns(self):
        from src.database.models import SensorAggregate1Hour
        col_names = {c.name for c in SensorAggregate1Hour.__table__.columns}
        required = {"bucket", "machine_id", "sensor_name", "avg_value", "min_value", "max_value", "sample_count"}
        missing = required - col_names
        assert not missing, f"SensorAggregate1Hour missing columns: {missing}"

    def test_mhi_hourly_has_required_columns(self):
        from src.database.models import MHIHourly
        col_names = {c.name for c in MHIHourly.__table__.columns}
        required = {"bucket", "machine_id", "avg_health_score", "avg_availability", "avg_reliability", "avg_condition"}
        missing = required - col_names
        assert not missing, f"MHIHourly missing columns: {missing}"

    def test_anomaly_rate_hourly_has_required_columns(self):
        from src.database.models import AnomalyRateHourly
        col_names = {c.name for c in AnomalyRateHourly.__table__.columns}
        required = {"bucket", "machine_id", "anomaly_count", "avg_anomaly_score"}
        missing = required - col_names
        assert not missing, f"AnomalyRateHourly missing columns: {missing}"

    def test_cagg_models_not_in_main_base_metadata(self):
        from src.database.models import Base
        table_names = set(Base.metadata.tables.keys())
        cagg_tables = {"cagg_sensor_15min", "cagg_sensor_1hour", "cagg_mhi_hourly", "cagg_anomaly_rate_hourly"}
        overlap = table_names & cagg_tables
        assert not overlap, f"CAGG models must not be in main Base metadata: {overlap}"


class TestInitDbIntegration:
    def test_no_fstring_sql_still_holds(self):
        content = _read_init_db()
        assert 'f"' not in content and "f'" not in content, (
            "init_db.py must not use f-strings for SQL queries"
        )

    def test_main_calls_all_compression_functions(self):
        content = _read_init_db()
        assert "configure_mhi_compression()" in content
        assert "configure_ai_act_compression()" in content
        assert "configure_anomaly_compression()" in content

    def test_main_calls_create_continuous_aggregates(self):
        content = _read_init_db()
        assert "create_continuous_aggregates()" in content

    def test_main_calls_configure_refresh_policies(self):
        content = _read_init_db()
        assert "configure_refresh_policies()" in content
