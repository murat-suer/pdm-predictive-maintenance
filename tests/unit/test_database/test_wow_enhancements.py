import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

BASELINE_PATH = Path(__file__).resolve().parents[3] / "alembic" / "versions" / "0001_baseline.py"
MODELS_PATH = Path(__file__).resolve().parents[3] / "src" / "database" / "models.py"
CONNECTION_PATH = Path(__file__).resolve().parents[3] / "src" / "database" / "connection.py"


def _read_baseline():
    return BASELINE_PATH.read_text(encoding="utf-8")


def _read_models():
    return MODELS_PATH.read_text(encoding="utf-8")


def _read_connection():
    return CONNECTION_PATH.read_text(encoding="utf-8")


class TestEnumTypes:
    def test_alarm_status_enum_in_migration(self):
        content = _read_baseline()
        assert "alarm_status_enum" in content

    def test_alarm_severity_enum_in_migration(self):
        content = _read_baseline()
        assert "severity_enum" in content

    def test_work_order_priority_enum_in_migration(self):
        content = _read_baseline()
        assert "work_order_priority_enum" in content

    def test_work_order_status_enum_in_migration(self):
        content = _read_baseline()
        assert "work_order_status_enum" in content

    def test_decision_action_enum_in_migration(self):
        content = _read_baseline()
        assert "decision_action_enum" in content

    def test_alarm_status_enum_values(self):
        content = _read_baseline()
        for val in ["NORMAL", "UNACKNOWLEDGED", "ACKNOWLEDGED", "NORMAL_UNACK", "SHELVED", "OUT_OF_SERVICE", "SUPPRESSED"]:
            assert f"'{val}'" in content, f"alarm_status_enum must contain '{val}'"

    def test_alarm_severity_enum_values(self):
        content = _read_baseline()
        assert "'WARNING'" in content
        assert "'CRITICAL'" in content

    def test_work_order_priority_enum_values(self):
        content = _read_baseline()
        for val in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            assert f"'{val}'" in content, f"work_order_priority_enum must contain '{val}'"

    def test_work_order_status_enum_values(self):
        content = _read_baseline()
        for val in ["PENDING", "IN_PROGRESS", "COMPLETED", "CANCELLED"]:
            assert f"'{val}'" in content, f"work_order_status_enum must contain '{val}'"

    def test_decision_action_enum_values(self):
        content = _read_baseline()
        for val in ["APPROVE", "DEFER", "REJECT"]:
            assert f"'{val}'" in content, f"decision_action_enum must contain '{val}'"

    def test_enum_create_with_checkfirst(self):
        content = _read_baseline()
        assert "CREATE TYPE" in content

    def test_models_use_pg_enum_alarm_status(self):
        content = _read_models()
        assert "alarm_status_enum" in content

    def test_models_use_pg_enum_alarm_severity(self):
        content = _read_models()
        assert "severity_enum" in content

    def test_models_use_pg_enum_work_order_priority(self):
        content = _read_models()
        assert "work_order_priority_enum" in content

    def test_models_use_pg_enum_work_order_status(self):
        content = _read_models()
        assert "work_order_status_enum" in content

    def test_decision_action_matches_migration_varchar(self):
        # Migration 0001 defines decision_log.action as VARCHAR(50) with
        # default 'PENDING'; a PG enum here would reject PENDING on read.
        content = _read_models()
        assert "decision_action_enum" not in content
        assert 'default="PENDING"' in content


class TestPartialIndexes:
    def test_active_alarms_partial_index_in_models(self):
        content = _read_models()
        assert "ix_alarm_state_active" in content

    def test_active_alarms_partial_index_filters_normal(self):
        content = _read_models()
        assert "NOT IN" in content or "not in" in content.lower()

    def test_active_alarms_partial_index_on_machine_id(self):
        from src.database.models import AlarmState
        idx_names = {idx.name for idx in AlarmState.__table__.indexes}
        assert "ix_alarm_state_active" in idx_names

    def test_pending_decisions_partial_index_in_models(self):
        content = _read_models()
        assert "ix_decision_log_pending" in content

    def test_pending_decisions_partial_index_on_decision_log(self):
        from src.database.models import DecisionLog
        idx_names = {idx.name for idx in DecisionLog.__table__.indexes}
        assert "ix_decision_log_pending" in idx_names

    def test_active_alarms_partial_index_in_migration(self):
        content = _read_baseline()
        assert "ix_alarm_state_active" in content

    def test_pending_decisions_partial_index_in_migration(self):
        content = _read_baseline()
        assert "ix_decision_log_pending" in content


class TestSchemaDocumentation:
    def test_sensor_readings_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE sensor_readings" in content

    def test_alarm_state_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE alarm_state" in content

    def test_anomaly_log_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE anomaly_log" in content

    def test_decision_log_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE decision_log" in content

    def test_work_orders_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE work_orders" in content

    def test_machine_baselines_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE machine_baselines" in content

    def test_machine_health_score_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE machine_health_score" in content

    def test_settings_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE settings" in content

    def test_maintenance_log_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE maintenance_log" in content

    def test_shift_reports_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE shift_reports" in content

    def test_ai_act_log_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE ai_act_log" in content

    def test_canary_probe_log_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE canary_probe_log" in content

    def test_alarm_state_transitions_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE alarm_state_transitions" in content

    def test_decision_audit_log_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE decision_audit_log" in content

    def test_decision_audit_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE decision_audit" in content

    def test_decision_work_orders_has_comment(self):
        content = _read_baseline()
        assert "COMMENT ON TABLE decision_work_orders" in content

    def test_column_comments_exist(self):
        content = _read_baseline()
        assert "COMMENT ON COLUMN" in content


class TestRetryWiring:
    def test_with_retry_function_exists(self):
        from src.database.connection import with_retry
        assert callable(with_retry)

    def test_with_retry_wraps_function(self):
        from src.database.connection import with_retry

        def sample_func():
            return 42

        wrapped = with_retry(sample_func)
        assert callable(wrapped)

    def test_with_retry_returns_result(self):
        from src.database.connection import with_retry

        def sample_func():
            return 42

        wrapped = with_retry(sample_func)
        assert wrapped() == 42

    def test_with_retry_retries_on_operational_error(self):
        from sqlalchemy.exc import OperationalError

        from src.database.connection import with_retry

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OperationalError("stmt", "params", Exception("connection lost"))
            return "success"

        wrapped = with_retry(flaky_func)
        result = wrapped()
        assert result == "success"
        assert call_count == 3

    def test_with_retry_reraises_after_max_attempts(self):
        from sqlalchemy.exc import OperationalError

        from src.database.connection import with_retry

        def always_fails():
            raise OperationalError("stmt", "params", Exception("connection lost"))

        wrapped = with_retry(always_fails)
        with pytest.raises(OperationalError):
            wrapped()
