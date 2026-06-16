"""Shared query helpers for dashboard routers."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy.orm import Session

from src.data_generator.machines import MACHINE_CONFIGS
from src.database.models import AlarmState, AnomalyLog, DecisionLog, MachineHealthScore, SensorReading

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


# Recommendation → status tier mapping.
# Primary signal: the most recent active DecisionLog recommendation.
# Unknown recommendation with an active alarm → "watch" (safe default).
_RECOMMENDATION_TO_STATUS: dict[str, str] = {
    "OBSERVE": "watch",
    "DISPATCH_TECHNICIAN": "watch",
    "PLANNED": "action",
    "REDUCE_LOAD": "critical",
    "SHUTDOWN": "critical",
}


# Tier severity order, for latching (a status never rewinds below the worst
# tier reached since the last repair).
_TIER_RANK: dict[str, int] = {"normal": 0, "watch": 1, "action": 2, "critical": 3}


def _last_repair_time(db: Session, machine_id: str) -> datetime | None:
    """When the machine was last actually repaired.

    A repair takes the machine down (downtime_minutes > 0) and is what resets
    the latched status. On-line technician inspections (no production stop)
    are part of the watch cycle and do NOT reset the latch.
    """
    from src.database.models import MaintenanceLog

    row = (
        db.query(MaintenanceLog.performed_at)
        .filter(
            MaintenanceLog.machine_id == machine_id,
            MaintenanceLog.performed_at.isnot(None),
            MaintenanceLog.downtime_minutes.isnot(None),
            MaintenanceLog.downtime_minutes > 0,
        )
        .order_by(MaintenanceLog.performed_at.desc())
        .first()
    )
    return as_utc(row[0]) if row is not None and row[0] is not None else None


def latched_status_tier(db: Session, machine_id: str) -> str:
    """Worst status tier the machine has reached since its last repair.

    ISA-18.2 latching: once a condition is flagged (e.g. OBSERVE → "watch"),
    the status does not silently fall back to normal just because the operator
    deferred it — it holds until a repair clears the condition, and only
    climbs if a more serious recommendation arrives.
    """
    repaired_at = _last_repair_time(db, machine_id)
    q = db.query(DecisionLog.ai_recommendation, DecisionLog.chosen_scenario_id).filter(
        DecisionLog.machine_id == machine_id
    )
    if repaired_at is not None:
        q = q.filter(DecisionLog.created_at > repaired_at)
    worst = "normal"
    for ai_rec, chosen in q.all():
        rec = ai_rec or chosen
        if rec is None:
            continue
        tier = _RECOMMENDATION_TO_STATUS.get(rec, "watch")
        if _TIER_RANK[tier] > _TIER_RANK[worst]:
            worst = tier
    return worst


def active_recommendation(db: Session, machine_id: str) -> str | None:
    """Return the recommendation string for the machine's most recent active decision.

    A decision is considered active when its linked alarm is still active
    (status in ACTIVE_ALARM_STATUSES) OR the decision itself is still PENDING.
    Returns ai_recommendation if present, else chosen_scenario_id, else None.
    """
    row = (
        db.query(DecisionLog)
        .join(AlarmState, AlarmState.id == DecisionLog.alarm_id, isouter=True)
        .filter(
            DecisionLog.machine_id == machine_id,
            (
                AlarmState.status.in_(ACTIVE_ALARM_STATUSES)
                | (DecisionLog.action == "PENDING")
            ),
        )
        .order_by(DecisionLog.created_at.desc())
        .first()
    )
    if row is None:
        return None
    return row.ai_recommendation or row.chosen_scenario_id


def derive_machine_status(db: Session, machine_id: str) -> tuple[str, str | None]:
    """Derive dashboard status: maintenance | offline | critical | action | watch | normal.

    Priority order:
    1. maintenance — machine is out of service.
    2. offline     — no recent sensor reading.
    3. recommendation-based tier — the most recent active decision recommendation
       wins outright over raw alarm severity.
    4. active alarm with no recommendation yet → "watch".
    5. normal      — no active anomaly or decision.
    """
    if is_machine_in_maintenance(db, machine_id):
        return "maintenance", None
    if is_machine_offline(db, machine_id):
        return "offline", None

    severity, fault_label = active_alarm_severity(db, machine_id)
    has_active_alarm = severity is not None

    # Latched: hold the worst tier reached since the last repair, so a deferred
    # OBSERVE ("watch") does not rewind to normal until maintenance clears it.
    tier = latched_status_tier(db, machine_id)
    # An active alarm with no recommendation yet is at least a watch.
    if tier == "normal" and has_active_alarm:
        tier = "watch"
    if tier == "normal":
        return "normal", None
    return tier, fault_label
