"""Shared query helpers for dashboard routers."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy.orm import Session

from src.data_generator.machines import MACHINE_CONFIGS
from src.database.models import AlarmState, AnomalyLog, MachineHealthScore, SensorReading

ACTIVE_ALARM_STATUSES = ("UNACKNOWLEDGED", "ACKNOWLEDGED", "NORMAL_UNACK", "SHELVED")
OFFLINE_AFTER_MINUTES = 5
MACHINE_TYPE_LABELS = {"AC": "Compressor", "HX": "Heat Exchanger", "CM": "Conveyor"}
# A health row without RUL (e.g. an MHI-only update) may briefly be the
# newest; fall back to the freshest row that does carry one within this window.
RUL_FRESHNESS_MINUTES = 30


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(dt: datetime) -> datetime:
    """Normalize naive timestamps (e.g. from SQLite) to UTC-aware."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def machine_type_label(machine_id: str) -> str:
    config = MACHINE_CONFIGS.get(machine_id, {})
    return MACHINE_TYPE_LABELS.get(config.get("type", ""), config.get("type", "Unknown"))


def latest_health_scores(db: Session, machine_ids: list[str]) -> dict[str, SimpleNamespace]:
    """Most recent health state per machine, with RUL coalesced.

    Returns detached snapshots (not ORM rows) so the coalesce never mutates
    a session-tracked object.
    """
    scores: dict[str, SimpleNamespace] = {}
    for machine_id in machine_ids:
        row = (
            db.query(MachineHealthScore)
            .filter(MachineHealthScore.machine_id == machine_id)
            .order_by(MachineHealthScore.calculated_at.desc())
            .first()
        )
        if row is None:
            continue
        snapshot = SimpleNamespace(
            health_score=row.health_score,
            availability_score=row.availability_score,
            reliability_score=row.reliability_score,
            condition_score=row.condition_score,
            classification=row.classification,
            rul_hours=row.rul_hours,
            confidence=row.confidence,
            calculated_at=row.calculated_at,
        )
        from src.database.models import MaintenanceLog

        # A completed repair invalidates every earlier RUL estimate — never
        # serve a pre-maintenance "4 hours left" on a machine that just came
        # back at full health.
        last_repair = (
            db.query(MaintenanceLog.performed_at)
            .filter(MaintenanceLog.machine_id == machine_id)
            .order_by(MaintenanceLog.performed_at.desc())
            .first()
        )
        repaired_at = (
            as_utc(last_repair[0])
            if last_repair is not None and last_repair[0] is not None
            else None
        )
        if repaired_at is not None and as_utc(snapshot.calculated_at) < repaired_at:
            # The overhaul renewed the machine: every pre-repair estimate -
            # not just RUL - describes a unit that no longer exists. Serve
            # honest dashes until the ML re-warms and writes a fresh row,
            # instead of haunting a rebuilt machine with 0% health.
            snapshot.rul_hours = None
            snapshot.confidence = None
            snapshot.health_score = None
            snapshot.availability_score = None
            snapshot.reliability_score = None
            snapshot.condition_score = None
            snapshot.classification = None
        if snapshot.rul_hours is None:
            cutoff = utc_now() - timedelta(minutes=RUL_FRESHNESS_MINUTES)
            if repaired_at is not None:
                cutoff = max(cutoff, repaired_at)
            rul_row = (
                db.query(MachineHealthScore)
                .filter(
                    MachineHealthScore.machine_id == machine_id,
                    MachineHealthScore.rul_hours.isnot(None),
                    MachineHealthScore.calculated_at >= cutoff,
                )
                .order_by(MachineHealthScore.calculated_at.desc())
                .first()
            )
            if rul_row is not None:
                snapshot.rul_hours = rul_row.rul_hours
                snapshot.confidence = rul_row.confidence
        scores[machine_id] = snapshot
    return scores


def health_history(db: Session, machine_id: str, points: int = 24) -> list[float]:
    """Last N health scores, oldest first."""
    rows = (
        db.query(MachineHealthScore.health_score)
        .filter(MachineHealthScore.machine_id == machine_id)
        .order_by(MachineHealthScore.calculated_at.desc())
        .limit(points)
        .all()
    )
    return [round(r[0] * 100.0, 1) for r in reversed(rows)]


def active_alarm_severity(db: Session, machine_id: str) -> tuple[str | None, str | None]:
    """Highest active alarm severity and its fault label for a machine.

    Returns (severity, fault_label); (None, None) when no active alarm exists.
    """
    rows = (
        db.query(AnomalyLog.severity, AnomalyLog.fault_type, AnomalyLog.top_contributing_sensor)
        .join(AlarmState, AlarmState.anomaly_id == AnomalyLog.id)
        .filter(
            AlarmState.machine_id == machine_id,
            AlarmState.status.in_(ACTIVE_ALARM_STATUSES),
        )
        .order_by(AlarmState.created_at.desc())
        .limit(10)
        .all()
    )
    if not rows:
        return None, None
    critical = next((r for r in rows if r[0] == "CRITICAL"), None)
    chosen = critical or rows[0]
    fault_label = chosen[1] or chosen[2]
    return chosen[0], fault_label


def is_machine_offline(db: Session, machine_id: str) -> bool:
    """A machine is offline when it produced no reading recently."""
    cutoff = utc_now() - timedelta(minutes=OFFLINE_AFTER_MINUTES)
    row = (
        db.query(SensorReading.timestamp)
        .filter(SensorReading.machine_id == machine_id, SensorReading.timestamp >= cutoff)
        .first()
    )
    return row is None


def is_machine_in_maintenance(db: Session, machine_id: str) -> bool:
    """True while an OUT_OF_SERVICE alarm has not been restored yet."""
    row = (
        db.query(AlarmState.id)
        .filter(
            AlarmState.machine_id == machine_id,
            AlarmState.status == "OUT_OF_SERVICE",
            AlarmState.oos_restored_at.is_(None),
        )
        .first()
    )
    return row is not None


# Health bands mirror the MHI classifier (mhi_calculator): Good >= 0.70,
# Critical < 0.55. A machine whose health has fallen into these bands must not
# show a green "normal" badge just because no discrete alarm has fired yet.
HEALTH_WARNING_BELOW = 0.70
HEALTH_CRITICAL_BELOW = 0.55


def _health_based_status(db: Session, machine_id: str) -> str | None:
    """Status implied by the latest MHI alone, or None if healthy / unknown."""
    row = (
        db.query(MachineHealthScore.health_score)
        .filter(MachineHealthScore.machine_id == machine_id)
        .order_by(MachineHealthScore.calculated_at.desc())
        .first()
    )
    if row is None or row[0] is None:
        return None
    health = row[0]
    if health < HEALTH_CRITICAL_BELOW:
        return "critical"
    if health < HEALTH_WARNING_BELOW:
        return "warning"
    return None


def derive_machine_status(db: Session, machine_id: str) -> tuple[str, str | None]:
    """Derive dashboard status (normal/warning/critical/maintenance/offline).

    Status is the worst of the active-alarm severity and the machine's health
    classification: degradation that has driven the MHI into the warning or
    critical band surfaces on the fleet view even before a discrete alarm fires.
    """
    if is_machine_in_maintenance(db, machine_id):
        return "maintenance", None
    if is_machine_offline(db, machine_id):
        return "offline", None
    severity, fault_label = active_alarm_severity(db, machine_id)
    if severity == "CRITICAL":
        return "critical", fault_label
    if severity == "WARNING":
        return "warning", fault_label
    health_status = _health_based_status(db, machine_id)
    if health_status is not None:
        return health_status, None
    return "normal", None
