import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func, text

Base = declarative_base()
CaggBase = declarative_base()


def utc_now():
    return datetime.now(UTC)


JSONType = JSON().with_variant(JSONB(), "postgresql")

# SQLite only auto-increments INTEGER primary keys; PostgreSQL keeps BIGINT.
BigIntPK = BigInteger().with_variant(Integer, "sqlite")

# ─── PostgreSQL ENUM Types (create_type=False — migration creates them) ──
ALARM_STATUS_ENUM = PG_ENUM(
    "NORMAL", "UNACKNOWLEDGED", "ACKNOWLEDGED", "NORMAL_UNACK",
    "SHELVED", "OUT_OF_SERVICE", "SUPPRESSED",
    name="alarm_status_enum", create_type=False,
)
SEVERITY_ENUM = PG_ENUM(
    "WARNING", "CRITICAL",
    name="severity_enum", create_type=False,
)
WORK_ORDER_PRIORITY_ENUM = PG_ENUM(
    "LOW", "MEDIUM", "HIGH", "CRITICAL",
    name="work_order_priority_enum", create_type=False,
)
WORK_ORDER_STATUS_ENUM = PG_ENUM(
    "PENDING", "IN_PROGRESS", "COMPLETED", "CANCELLED",
    name="work_order_status_enum", create_type=False,
)


# ─── Table 1: sensor_readings ─────────────────────────────────────────────
class SensorReading(Base):
    __tablename__ = "sensor_readings"

    machine_id = Column(String(20), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    sensor_name = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, default=False, nullable=False)
    anomaly_score = Column(Float, nullable=True)
    upstream_effect = Column(Boolean, default=False)
    machine_phase = Column(String(20), nullable=True)
    fft_data = Column(JSONType, nullable=True)
    present = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("machine_id", "sensor_name", "timestamp"),
        Index("ix_sensor_readings_machine_time", "machine_id", "timestamp"),
        Index(
            "ix_sensor_readings_anomaly",
            "machine_id",
            "is_anomaly",
            postgresql_where=text("is_anomaly = true"),
        ),
    )


# ─── Table 2: machine_baselines ───────────────────────────────────────────
class MachineBaseline(Base):
    __tablename__ = "machine_baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String(20), nullable=False, index=True)
    sensor = Column(String(50), nullable=False)
    mean_value = Column(Float, nullable=False)
    std_value = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False)
    calibrated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=utc_now, server_default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)

    __table_args__ = (UniqueConstraint("machine_id", "sensor", "is_active", name="uq_active_baseline"),)


# ─── Table 3: anomaly_log ─────────────────────────────────────────────────
class AnomalyLog(Base):
    __tablename__ = "anomaly_log"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    machine_id = Column(String(20), nullable=False, index=True)
    detected_at = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    shap_values = Column(JSONType, nullable=True)
    top_contributing_sensor = Column(String(50), nullable=True)
    severity = Column(
        SEVERITY_ENUM,
        nullable=False,
    )
    status = Column(String(20), nullable=False, default="ACTIVE")
    resolved_at = Column(TIMESTAMP(timezone=True), nullable=True)
    resolution_type = Column(String(30), nullable=True)
    upstream_effect = Column(Boolean, default=False)
    fault_type = Column(String(50), nullable=True)
    fault_confidence = Column(Float, nullable=True)

    alarms = relationship("AlarmState", back_populates="anomaly")

    __table_args__ = (
        CheckConstraint("severity IN ('WARNING', 'CRITICAL')", name="ck_anomaly_log_severity"),
    )


# ─── Table 4: alarm_state ────────────────────────────────────────────────
class AlarmState(Base):
    __tablename__ = "alarm_state"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    anomaly_id = Column(BigInteger, ForeignKey("anomaly_log.id"), nullable=False, index=True)
    machine_id = Column(String(20), nullable=False, index=True)
    level = Column(Integer, nullable=False)
    status = Column(
        ALARM_STATUS_ENUM,
        nullable=False, default="UNACKNOWLEDGED",
    )
    created_at = Column(TIMESTAMP(timezone=True), nullable=False)
    last_updated = Column(TIMESTAMP(timezone=True), nullable=False)

    trend_direction = Column(String(15), nullable=True)
    trend_check_count = Column(Integer, default=0)

    escalated_at = Column(TIMESTAMP(timezone=True), nullable=True)

    shelved_at = Column(TIMESTAMP(timezone=True), nullable=True)
    shelved_until = Column(TIMESTAMP(timezone=True), nullable=True)
    shelve_reason = Column(Text, nullable=True)
    shelved_by_role = Column(String(30), nullable=True)

    oos_at = Column(TIMESTAMP(timezone=True), nullable=True)
    work_order_id = Column(String(50), nullable=True)
    oos_restored_at = Column(TIMESTAMP(timezone=True), nullable=True)

    suppressed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    suppressed_by = Column(String(30), nullable=True)
    suppress_reason = Column(Text, nullable=True)
    suppress_review_due = Column(TIMESTAMP(timezone=True), nullable=True)

    anomaly = relationship("AnomalyLog", back_populates="alarms")
    decisions = relationship("DecisionLog", back_populates="alarm")
    transitions = relationship("AlarmStateTransition", back_populates="alarm")
    maintenance = relationship("MaintenanceLog", back_populates="alarm")

    __table_args__ = (
        CheckConstraint(
            "status IN ('NORMAL', 'UNACKNOWLEDGED', 'ACKNOWLEDGED', 'NORMAL_UNACK', 'SHELVED', 'OUT_OF_SERVICE', 'SUPPRESSED')",
            name="ck_alarm_state_status",
        ),
        Index(
            "ix_alarm_state_active",
            "machine_id", "created_at",
            postgresql_where=text("status NOT IN ('NORMAL', 'SUPPRESSED', 'OUT_OF_SERVICE')"),
        ),
    )


# ─── Table 5: alarm_state_transitions ───────────────────────────────────
class AlarmStateTransition(Base):
    __tablename__ = "alarm_state_transitions"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    alarm_id = Column(BigInteger, ForeignKey("alarm_state.id"), nullable=False, index=True)
    from_state = Column(String(25), nullable=False)
    to_state = Column(String(25), nullable=False)
    operator_role = Column(String(30), nullable=True)
    reason = Column(Text, nullable=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    alarm = relationship("AlarmState", back_populates="transitions")

    __table_args__ = (
        Index("ix_alarm_transitions_timestamp", "timestamp"),
    )


# ─── Table 6: decision_log ───────────────────────────────────────────────
class DecisionLog(Base):
    __tablename__ = "decision_log"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    alarm_id = Column(BigInteger, ForeignKey("alarm_state.id"), nullable=True, index=True)
    machine_id = Column(String(20), nullable=False)
    # Migration 0001 defines this as VARCHAR(50) with default 'PENDING';
    # PENDING → APPROVE/DEFER/REJECT is enforced in application logic.
    action = Column(String(50), nullable=False, default="PENDING")
    scenario_id = Column(String(32), nullable=True)
    chosen_scenario_id = Column(String(32), nullable=True)
    operator_role = Column(String(30), nullable=True)
    decided_by = Column(String(30), nullable=True)
    decided_at = Column(TIMESTAMP(timezone=True), nullable=True)
    response_time_s = Column(Integer, nullable=True)
    ai_recommendation = Column(String(32), nullable=True)
    overridden = Column(Boolean, default=False)
    escalation_level = Column(Integer, default=1)
    auto_approved = Column(Boolean, default=False)
    resolution_source = Column(String(30), nullable=True)
    outcome = Column(String(30), nullable=True)
    due_at = Column(TIMESTAMP(timezone=True), nullable=True)
    scenarios_presented = Column(JSONType, nullable=True)
    notes = Column(Text, nullable=True)
    downtime_minutes = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=utc_now, server_default=func.now())

    alarm = relationship("AlarmState", back_populates="decisions")
    audit_logs = relationship("DecisionAuditLog", back_populates="decision")

    __table_args__ = (
        Index("ix_audit_machine_time", "alarm_id", "created_at"),
        Index(
            "ix_decision_log_pending",
            "machine_id", "created_at",
            postgresql_where=text("action = 'PENDING'"),
        ),
    )


# ─── Table 6b: decision_audit_log ────────────────────────────────────────
class DecisionAuditLog(Base):
    __tablename__ = "decision_audit_log"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    decision_id = Column(String(36), ForeignKey("decision_log.id"), nullable=False, index=True)
    alarm_id = Column(BigInteger, ForeignKey("alarm_state.id"), nullable=True, index=True)
    action = Column(String(50), nullable=False)
    scenario_id = Column(String(32), nullable=True)
    operator_role = Column(String(30), nullable=True)
    response_time_s = Column(Integer, nullable=True)
    ai_recommendation = Column(String(32), nullable=True)
    overridden = Column(Boolean, default=False)
    override_reason = Column(Text, nullable=True)
    escalation_level = Column(Integer, default=1)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=utc_now, server_default=func.now())

    decision = relationship("DecisionLog", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_decision_audit_decision_id", "decision_id"),
        Index("ix_decision_audit_created_at", "created_at"),
    )


# ─── Table 7: machine_health_score ──────────────────────────────────────
class MachineHealthScore(Base):
    __tablename__ = "machine_health_score"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    machine_id = Column(String(20), nullable=False)
    calculated_at = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    health_score = Column(Float, nullable=False)
    availability_score = Column(Float, nullable=True)
    reliability_score = Column(Float, nullable=True)
    condition_score = Column(Float, nullable=True)
    rul_hours = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    classification = Column(String(30), nullable=True)

    __table_args__ = (
        Index("ix_mhi_machine_time", "machine_id", "calculated_at"),
        CheckConstraint("health_score >= 0.0 AND health_score <= 1.0", name="ck_machine_health_score_range"),
    )


# ─── Table 8: canary_probe_log ───────────────────────────────────────────
class CanaryProbeLog(Base):
    __tablename__ = "canary_probe_log"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    probe_id = Column(String(36), nullable=False, unique=True)
    machine_id = Column(String(20), nullable=False)
    probe_type = Column(String(10), nullable=False)
    scenario = Column(String(100), nullable=True)
    started_at = Column(TIMESTAMP(timezone=True), nullable=False)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    detected = Column(Boolean, nullable=True)
    expected = Column(Boolean, nullable=False, default=True)
    success = Column(Boolean, nullable=True)
    duration_s = Column(Integer, nullable=True)
    triggered_by = Column(String(30), nullable=False)
    recalibration_triggered = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)

    __table_args__ = (Index("ix_canary_machine_time", "machine_id", "started_at"),)


# ─── Table 9: settings ───────────────────────────────────────────────────
class Settings(Base):
    __tablename__ = "settings"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    category = Column(String(20), nullable=False)
    key = Column(String(60), nullable=False)
    value = Column(String(255), nullable=False)
    unit = Column(String(20), nullable=True)
    description = Column(Text, nullable=True)
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    updated_by = Column(String(30), nullable=True)

    __table_args__ = (UniqueConstraint("category", "key", name="uq_settings_category_key"),)


# ─── Table 10: maintenance_log ───────────────────────────────────────────
class MaintenanceLog(Base):
    __tablename__ = "maintenance_log"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    machine_id = Column(String(20), nullable=False)
    alarm_id = Column(BigInteger, ForeignKey("alarm_state.id"), nullable=True, index=True)
    scheduled_at = Column(TIMESTAMP(timezone=True), nullable=True)
    performed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    technician_notes = Column(Text, nullable=True)
    fault_found = Column(Boolean, nullable=True)
    fault_description = Column(Text, nullable=True)
    downtime_minutes = Column(Integer, nullable=True)
    cost_eur = Column(Float, nullable=True)

    alarm = relationship("AlarmState", back_populates="maintenance")

    __table_args__ = (Index("ix_maintenance_machine_time", "machine_id", "performed_at"),)


# ─── Table 11: work_orders ───────────────────────────────────────────────
class WorkOrder(Base):
    __tablename__ = "work_orders"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    machine_id = Column(String(20), nullable=False)
    decision_id = Column(String(36), ForeignKey("decision_log.id"), nullable=True)
    alarm_id = Column(BigInteger, ForeignKey("alarm_state.id"), nullable=True)
    fault_type = Column(String(50), nullable=True)
    recommended_action = Column(Text, nullable=True)
    action_type = Column(String(50), nullable=True)
    priority = Column(
        WORK_ORDER_PRIORITY_ENUM,
        nullable=False, default="MEDIUM",
    )
    estimated_cost_eur = Column(Float, nullable=True)
    assigned_team = Column(String(50), nullable=True)
    status = Column(
        WORK_ORDER_STATUS_ENUM,
        nullable=False, default="OPEN",
    )
    created_by = Column(String(50), nullable=True)
    scenario_id = Column(String(32), nullable=True)
    work_order_number = Column(String(30), nullable=True, unique=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=utc_now, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=utc_now, server_default=func.now(), onupdate=func.now())

    decision = relationship("DecisionLog", backref="work_orders")
    alarm = relationship("AlarmState", backref="work_orders")

    __table_args__ = (
        Index("ix_work_orders_machine_id", "machine_id"),
        Index("ix_work_orders_decision_id", "decision_id"),
        Index("ix_work_orders_alarm_id", "alarm_id"),
        Index("ix_work_orders_status", "status"),
        Index("ix_work_orders_created_at", "created_at"),
    )


# ─── Table 12: shift_reports ─────────────────────────────────────────────
class ShiftReport(Base):
    __tablename__ = "shift_reports"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    shift_start = Column(TIMESTAMP(timezone=True), nullable=False)
    shift_end = Column(TIMESTAMP(timezone=True), nullable=False)
    shift_type = Column(String(10), nullable=False)
    generated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    report_data = Column(JSONType, nullable=False)
    html_content = Column(Text, nullable=True)

    __table_args__ = (Index("ix_shift_reports_start", "shift_start"),)


# ─── Table 13: ai_act_log ────────────────────────────────────────────────
class AIActLog(Base):
    __tablename__ = "ai_act_log"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    machine_id = Column(String(20), nullable=False)
    model_version = Column(String(30), nullable=False)
    decision_type = Column(String(50), nullable=False)
    features_snapshot = Column(JSONType, nullable=True)
    output = Column(JSONType, nullable=True)
    shap_values = Column(JSONType, nullable=True)
    action_taken = Column(String(100), nullable=True)
    human_action = Column(String(100), nullable=True)

    __table_args__ = (Index("ix_ai_act_log_machine_time", "machine_id", "timestamp"),)


# ─── Table 14: decision_audit (orphan from migration a7b9c2d4f1e3) ──────
class DecisionAudit(Base):
    __tablename__ = "decision_audit"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    machine_id = Column(String(50), nullable=False)
    alarm_id = Column(String(50), nullable=False)
    state_from = Column(String(30), nullable=False)
    state_to = Column(String(30), nullable=False)
    ml_inputs = Column(JSONB(), nullable=True)
    diagnosis = Column(JSONB(), nullable=True)
    options_offered = Column(JSONB(), nullable=True)
    recommended_option_id = Column(String(30), nullable=True)
    operator_choice = Column(String(30), nullable=True)
    operator_id = Column(String(50), nullable=True)
    operator_type = Column(String(10), nullable=False)
    decision_source = Column(String(20), nullable=False)
    operator_reason = Column(Text, nullable=True)
    severity = Column(String(10), nullable=True)
    time_to_action_minutes = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_decision_audit_machine_time", "machine_id", text("timestamp DESC")),
        Index("idx_decision_audit_alarm_time", "alarm_id", text("timestamp DESC")),
    )


# ─── Table 15: decision_work_orders (orphan from migration b8c0d3e5f2a4) ─
class DecisionWorkOrder(Base):
    __tablename__ = "decision_work_orders"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    machine_id = Column(String(50), nullable=False)
    alarm_id = Column(String(50), nullable=True)
    decision_audit_id = Column(BigInteger, ForeignKey("decision_audit.id", ondelete="SET NULL"), nullable=True)
    work_order_type = Column(String(30), nullable=False)
    priority = Column(String(10), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    assigned_to = Column(String(100), nullable=True)
    created_by = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, server_default=text("'open'"))
    estimated_cost_saved = Column(Numeric(12, 2), nullable=True)
    closed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    close_notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_decision_work_orders_machine_time", "machine_id", text("timestamp DESC")),
        Index("idx_decision_work_orders_status_priority", "status", "priority"),
    )


# ─── Continuous Aggregate Models (Read-Only) ─────────────────────────────
class SensorAggregate15Min(CaggBase):
    __tablename__ = "cagg_sensor_15min"
    __table_args__ = {"info": {"materialized_view": True}}

    bucket = Column(TIMESTAMP(timezone=True), primary_key=True)
    machine_id = Column(String(20), primary_key=True)
    sensor_name = Column(String(50), primary_key=True)
    avg_value = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    stddev_value = Column(Float)
    sample_count = Column(BigInteger)


class SensorAggregate1Hour(CaggBase):
    __tablename__ = "cagg_sensor_1hour"
    __table_args__ = {"info": {"materialized_view": True}}

    bucket = Column(TIMESTAMP(timezone=True), primary_key=True)
    machine_id = Column(String(20), primary_key=True)
    sensor_name = Column(String(50), primary_key=True)
    avg_value = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    sample_count = Column(BigInteger)


class MHIHourly(CaggBase):
    __tablename__ = "cagg_mhi_hourly"
    __table_args__ = {"info": {"materialized_view": True}}

    bucket = Column(TIMESTAMP(timezone=True), primary_key=True)
    machine_id = Column(String(20), primary_key=True)
    avg_health_score = Column(Float)
    avg_availability = Column(Float)
    avg_reliability = Column(Float)
    avg_condition = Column(Float)


class AnomalyRateHourly(CaggBase):
    __tablename__ = "cagg_anomaly_rate_hourly"
    __table_args__ = {"info": {"materialized_view": True}}

    bucket = Column(TIMESTAMP(timezone=True), primary_key=True)
    machine_id = Column(String(20), primary_key=True)
    anomaly_count = Column(BigInteger)
    avg_anomaly_score = Column(Float)


# ─── SQLAlchemy Event Listeners ──────────────────────────────────────────
_AUTO_APPROVE_DELAY_SECONDS = int(os.getenv("AUTO_APPROVE_DELAY_SECONDS", "180"))


def _default_due_at():
    """Human response window: base delay with ±20% jitter, so the
    simulated-operator takeover does not fire in lockstep."""
    import random

    jitter = random.uniform(0.8, 1.2)
    return datetime.now(UTC) + timedelta(seconds=_AUTO_APPROVE_DELAY_SECONDS * jitter)


@event.listens_for(AlarmState, "after_insert")
def _ensure_decision_log_on_alarm(mapper, connection, target):
    if target.status != "UNACKNOWLEDGED":
        return
    from sqlalchemy import select as _select

    existing = connection.execute(
        _select(DecisionLog).where(DecisionLog.alarm_id == target.id)
    ).first()
    if existing:
        return
    now_utc = datetime.now(UTC)
    connection.execute(
        DecisionLog.__table__.insert().values(
            id=str(uuid4()),
            alarm_id=target.id,
            machine_id=target.machine_id,
            action="PENDING",
            escalation_level=target.level,
            created_at=now_utc,
            due_at=_default_due_at(),
        )
    )
