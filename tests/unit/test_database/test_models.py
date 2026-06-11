import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestAllModelsExist:
    def test_sensor_reading_model(self):
        from src.database.models import SensorReading
        assert SensorReading.__tablename__ == "sensor_readings"

    def test_anomaly_log_model(self):
        from src.database.models import AnomalyLog
        assert AnomalyLog.__tablename__ == "anomaly_log"

    def test_alarm_state_model(self):
        from src.database.models import AlarmState
        assert AlarmState.__tablename__ == "alarm_state"

    def test_alarm_state_transition_model(self):
        from src.database.models import AlarmStateTransition
        assert AlarmStateTransition.__tablename__ == "alarm_state_transitions"

    def test_decision_log_model(self):
        from src.database.models import DecisionLog
        assert DecisionLog.__tablename__ == "decision_log"

    def test_decision_audit_log_model(self):
        from src.database.models import DecisionAuditLog
        assert DecisionAuditLog.__tablename__ == "decision_audit_log"

    def test_machine_health_score_model(self):
        from src.database.models import MachineHealthScore
        assert MachineHealthScore.__tablename__ == "machine_health_score"

    def test_canary_probe_log_model(self):
        from src.database.models import CanaryProbeLog
        assert CanaryProbeLog.__tablename__ == "canary_probe_log"

    def test_settings_model(self):
        from src.database.models import Settings
        assert Settings.__tablename__ == "settings"

    def test_maintenance_log_model(self):
        from src.database.models import MaintenanceLog
        assert MaintenanceLog.__tablename__ == "maintenance_log"

    def test_work_order_model(self):
        from src.database.models import WorkOrder
        assert WorkOrder.__tablename__ == "work_orders"

    def test_shift_report_model(self):
        from src.database.models import ShiftReport
        assert ShiftReport.__tablename__ == "shift_reports"

    def test_ai_act_log_model(self):
        from src.database.models import AIActLog
        assert AIActLog.__tablename__ == "ai_act_log"

    def test_machine_baseline_model(self):
        from src.database.models import MachineBaseline
        assert MachineBaseline.__tablename__ == "machine_baselines"

    def test_decision_audit_model(self):
        from src.database.models import DecisionAudit
        assert DecisionAudit.__tablename__ == "decision_audit"

    def test_decision_work_orders_model(self):
        from src.database.models import DecisionWorkOrder
        assert DecisionWorkOrder.__tablename__ == "decision_work_orders"


class TestModelCount:
    def test_total_model_count(self):
        from src.database.models import Base
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "sensor_readings",
            "anomaly_log",
            "alarm_state",
            "alarm_state_transitions",
            "decision_log",
            "decision_audit_log",
            "machine_health_score",
            "canary_probe_log",
            "settings",
            "maintenance_log",
            "work_orders",
            "shift_reports",
            "ai_act_log",
            "machine_baselines",
            "decision_audit",
            "decision_work_orders",
        }
        assert table_names == expected, (
            f"Expected 16 tables, got {len(table_names)}: {table_names}"
        )


class TestCheckConstraints:
    def test_alarm_status_constraint_exists(self):
        from sqlalchemy import CheckConstraint

        from src.database.models import AlarmState
        constraints = [
            c for c in AlarmState.__table__.constraints
            if isinstance(c, CheckConstraint)
        ]
        assert len(constraints) >= 1, "AlarmState must have a CHECK constraint on status"
        constraint_sql = str(constraints[0].sqltext.compile(compile_kwargs={"literal_binds": True}))
        assert "status" in constraint_sql.lower() or "UNACKNOWLEDGED" in constraint_sql

    def test_anomaly_severity_constraint_exists(self):
        from sqlalchemy import CheckConstraint

        from src.database.models import AnomalyLog
        constraints = [
            c for c in AnomalyLog.__table__.constraints
            if isinstance(c, CheckConstraint)
        ]
        assert len(constraints) >= 1, "AnomalyLog must have a CHECK constraint on severity"

    def test_health_score_constraint_exists(self):
        from sqlalchemy import CheckConstraint

        from src.database.models import MachineHealthScore
        constraints = [
            c for c in MachineHealthScore.__table__.constraints
            if isinstance(c, CheckConstraint)
        ]
        assert len(constraints) >= 1, "MachineHealthScore must have a CHECK constraint on health_score"


class TestIndexes:
    def test_anomaly_log_has_machine_id_index(self):
        from src.database.models import AnomalyLog
        columns_with_index = {
            c.name for c in AnomalyLog.__table__.columns if c.index
        }
        index_columns = set()
        for idx in AnomalyLog.__table__.indexes:
            for col in idx.columns:
                index_columns.add(col.name)
        all_indexed = columns_with_index | index_columns
        assert "machine_id" in all_indexed, "anomaly_log.machine_id must be indexed"

    def test_alarm_state_has_machine_id_index(self):
        from src.database.models import AlarmState
        columns_with_index = {
            c.name for c in AlarmState.__table__.columns if c.index
        }
        index_columns = set()
        for idx in AlarmState.__table__.indexes:
            for col in idx.columns:
                index_columns.add(col.name)
        all_indexed = columns_with_index | index_columns
        assert "machine_id" in all_indexed, "alarm_state.machine_id must be indexed"

    def test_alarm_state_transitions_no_duplicate_index(self):
        from src.database.models import AlarmStateTransition
        alarm_id_indexes = []
        for idx in AlarmStateTransition.__table__.indexes:
            col_names = {c.name for c in idx.columns}
            if col_names == {"alarm_id"}:
                alarm_id_indexes.append(idx.name)
        assert len(alarm_id_indexes) == 1, (
            f"alarm_state_transitions should have exactly 1 index on alarm_id, "
            f"found {len(alarm_id_indexes)}: {alarm_id_indexes}"
        )


class TestDecisionAuditModel:
    def test_has_required_columns(self):
        from src.database.models import DecisionAudit
        col_names = {c.name for c in DecisionAudit.__table__.columns}
        required = {
            "id", "timestamp", "machine_id", "alarm_id",
            "state_from", "state_to", "operator_type", "decision_source",
        }
        missing = required - col_names
        assert not missing, f"DecisionAudit missing columns: {missing}"

    def test_has_indexes(self):
        from src.database.models import DecisionAudit
        idx_names = {idx.name for idx in DecisionAudit.__table__.indexes}
        assert "idx_decision_audit_machine_time" in idx_names
        assert "idx_decision_audit_alarm_time" in idx_names


class TestDecisionWorkOrdersModel:
    def test_has_required_columns(self):
        from src.database.models import DecisionWorkOrder
        col_names = {c.name for c in DecisionWorkOrder.__table__.columns}
        required = {
            "id", "timestamp", "machine_id", "decision_audit_id",
            "work_order_type", "priority", "status",
        }
        missing = required - col_names
        assert not missing, f"DecisionWorkOrder missing columns: {missing}"

    def test_has_indexes(self):
        from src.database.models import DecisionWorkOrder
        idx_names = {idx.name for idx in DecisionWorkOrder.__table__.indexes}
        assert "idx_decision_work_orders_machine_time" in idx_names
        assert "idx_decision_work_orders_status_priority" in idx_names

    def test_has_fk_to_decision_audit(self):
        from src.database.models import DecisionWorkOrder
        fk_targets = set()
        for col in DecisionWorkOrder.__table__.columns:
            for fk in col.foreign_keys:
                fk_targets.add(fk.target_fullname)
        assert "decision_audit.id" in fk_targets, (
            "DecisionWorkOrder must have FK to decision_audit.id"
        )
