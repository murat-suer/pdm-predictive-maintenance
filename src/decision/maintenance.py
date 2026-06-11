"""Maintenance lifecycle: schedule, execute, complete, account savings.

When a maintenance-bearing scenario (REDUCE_LOAD / PLANNED / SHUTDOWN) is
approved, a job lands in the Redis sorted-set schedule. The decision runner
polls it and drives the loop:

  start:    PAUSE_LINE → alarm OUT_OF_SERVICE → work order IN_PROGRESS
  complete: RESET_MACHINE (full health) + RESUME_LINE + MAINTENANCE_DONE
            → anomaly RESOLVED, alarm NORMAL, work order COMPLETED,
            maintenance_log row with actual cost and downtime.

Durations are simulation hours (the demo runs at SIMULATION_SPEED× wall
clock): a 2 sim-hour repair takes ~14 s of real time at 500×. Savings are
the run-to-failure counterfactual recorded on the decision (notes JSON)
minus what the planned intervention actually cost.
"""
import json
import logging
from datetime import UTC, datetime

from src.config import settings
from src.data_generator.control import machine_line, publish_control

logger = logging.getLogger(__name__)

SCHEDULE_KEY = "maintenance_schedule"

# Simulation-time durations (hours). PLANNED waits for the next shift
# boundary; SHUTDOWN stops the line immediately.
PLANNED_LEAD_SIM_HOURS = 4.0
MAINTENANCE_SIM_HOURS = 2.0
REDUCE_LOAD_FACTOR = 0.8
SHIFT_CHANGE_OVERLAP_HOURS = 0.5

MAINTENANCE_SCENARIOS = {"REDUCE_LOAD", "PLANNED", "SHUTDOWN"}

# Reactive (run-to-failure) repair takes much longer than planned work:
# crew call-out, diagnosis on a dead machine, parts logistics, repair.
EMERGENCY_SIM_HOURS = 9.0
MAINTENANCE_LOCK_PREFIX = "maintenance:lock:"
MAINTENANCE_LOCK_TTL_S = 1800


def _sim_hours_to_real_seconds(hours: float, cap_s: float | None = None) -> float:
    """Convert sim-hours to real seconds, optionally capped.

    The cap keeps the demo loop watchable at low simulation speeds: the
    narrative duration stays in sim-hours, but a visitor never waits more
    than a couple of real minutes to see the line stop or recover.
    """
    speed = max(float(settings.SIMULATION_SPEED), 1.0)
    real = hours * 3600.0 / speed
    return min(real, cap_s) if cap_s is not None else real


def _planned_cost(decision) -> float:
    chosen = decision.chosen_scenario_id
    for item in decision.scenarios_presented or []:
        if item.get("scenario") == chosen:
            return float(item.get("cost", 0.0))
    return 0.0


def _decision_context(decision) -> dict:
    try:
        return json.loads(decision.notes) if decision.notes else {}
    except (TypeError, ValueError):
        return {}


def _acquire_lock(r, machine_id: str) -> bool:
    """One maintenance pipeline per machine at a time."""
    try:
        return bool(
            r.set(
                MAINTENANCE_LOCK_PREFIX + machine_id,
                "1",
                nx=True,
                ex=MAINTENANCE_LOCK_TTL_S,
            )
        )
    except Exception:
        return True  # if Redis misbehaves, do not deadlock the loop


def _release_lock(r, machine_id: str) -> None:
    try:
        r.delete(MAINTENANCE_LOCK_PREFIX + machine_id)
    except Exception:
        pass


def schedule_from_decision(db, r, decision, alarm) -> None:
    """Queue the maintenance job implied by an approved scenario."""
    scenario = decision.chosen_scenario_id
    if scenario not in MAINTENANCE_SCENARIOS:
        return
    if not _acquire_lock(r, decision.machine_id):
        logger.info(f"Maintenance already in flight for {decision.machine_id}; skipping")
        return

    now = datetime.now(UTC).timestamp()
    if scenario == "SHUTDOWN":
        start_at = now  # emergency: stop the line immediately
    else:
        start_at = now + _sim_hours_to_real_seconds(
            PLANNED_LEAD_SIM_HOURS, cap_s=float(settings.MAINT_MAX_REAL_LEAD_S)
        )

    if scenario == "REDUCE_LOAD":
        publish_control(r, "SET_LOAD", machine_id=decision.machine_id, factor=REDUCE_LOAD_FACTOR)

    job = {
        "phase": "START",
        "machine_id": decision.machine_id,
        "decision_id": decision.id,
        "alarm_id": alarm.id,
        "scenario": scenario,
    }
    try:
        r.zadd(SCHEDULE_KEY, {json.dumps(job): start_at})
        logger.info(
            f"Maintenance scheduled: {decision.machine_id} {scenario} "
            f"starts in {max(start_at - now, 0):.0f}s"
        )
    except Exception as exc:
        logger.error(f"Maintenance scheduling failed: {exc}")

    _ensure_work_order(db, decision, alarm)


def _ensure_work_order(db, decision, alarm) -> None:
    from src.database.models import WorkOrder

    existing = (
        db.query(WorkOrder).filter(WorkOrder.decision_id == decision.id).first()
    )
    if existing is not None:
        return
    anomaly = alarm.anomaly if alarm is not None else None
    severity = anomaly.severity if anomaly is not None else "WARNING"
    work_order = WorkOrder(
        machine_id=decision.machine_id,
        decision_id=decision.id,
        alarm_id=alarm.id if alarm is not None else None,
        fault_type=anomaly.fault_type if anomaly is not None else None,
        recommended_action=decision.chosen_scenario_id,
        action_type=decision.chosen_scenario_id,
        priority="CRITICAL" if severity == "CRITICAL" else "HIGH",
        estimated_cost_eur=_planned_cost(decision),
        status="PENDING",
        created_by=decision.decided_by,
        scenario_id=decision.chosen_scenario_id,
        work_order_number=alarm.work_order_id
        if alarm is not None and alarm.work_order_id
        else f"WO-{decision.machine_id}-{datetime.now(UTC).strftime('%H%M%S')}",
    )
    db.add(work_order)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(f"Work order creation failed: {exc}")


OBSERVATION_SCENARIOS = {"OBSERVE", "DISPATCH_TECHNICIAN"}
DISPATCH_INSPECTION_COST_EUR = 150.0


def close_observation_cycle(db, decision, alarm) -> None:
    """OBSERVE / DISPATCH_TECHNICIAN are point-in-time decisions, not a
    permanent dismissal.

    Closing the anomaly/alarm cycle lets the ML pipeline raise a fresh
    alarm (with an updated, lower RUL) as degradation continues — so the
    next decision is made on current data instead of never being asked.
    A technician dispatch additionally books the call-out cost in the
    maintenance ledger (inspection only — the line keeps running).
    """
    from src.database.models import AlarmStateTransition, AnomalyLog

    if decision.chosen_scenario_id not in OBSERVATION_SCENARIOS or alarm is None:
        return
    dispatched = decision.chosen_scenario_id == "DISPATCH_TECHNICIAN"
    now = datetime.now(UTC)
    db.query(AnomalyLog).filter(
        AnomalyLog.machine_id == decision.machine_id,
        AnomalyLog.status == "ACTIVE",
    ).update(
        {
            "status": "RESOLVED",
            "resolved_at": now,
            "resolution_type": "TECHNICIAN_DISPATCHED" if dispatched else "OBSERVE_REVIEW",
        },
        synchronize_session=False,
    )
    if dispatched:
        _book_inspection(db, decision, now)
    db.add(
        AlarmStateTransition(
            alarm_id=alarm.id,
            from_state=alarm.status,
            to_state="NORMAL",
            operator_role=decision.operator_role,
            reason=(
                "DISPATCH_TECHNICIAN: on-line inspection ordered; alarm re-raised if degradation persists"
                if dispatched
                else "OBSERVE: monitoring continues; alarm re-raised if degradation persists"
            ),
            timestamp=now,
        )
    )
    alarm.status = "NORMAL"
    try:
        db.commit()
    except Exception:
        db.rollback()


def _book_inspection(db, decision, now) -> None:
    """Ledger entry for an on-line technician inspection (no downtime)."""
    from src.database.models import MaintenanceLog

    db.add(
        MaintenanceLog(
            machine_id=decision.machine_id,
            alarm_id=decision.alarm_id,
            scheduled_at=now,
            performed_at=now,
            technician_notes=(
                "On-line inspection after repeated OBSERVE — fault could not be "
                "classified from sensor signatures; technician dispatched while "
                "the line keeps running."
            ),
            fault_found=False,
            downtime_minutes=0,
            cost_eur=DISPATCH_INSPECTION_COST_EUR,
        )
    )


def schedule_emergency_repairs(db, r) -> int:
    """Safety net: a machine that reached FAILED gets a reactive repair.

    This is the run-to-failure path the PdM system exists to avoid — the
    crew is called out after the fact, so the repair takes EMERGENCY_SIM_HOURS
    and is booked at the full reactive cost (no savings).
    """

    from src.data_generator.machines import MACHINE_CONFIGS
    from src.database.models import SensorReading

    scheduled = 0
    now = datetime.now(UTC)
    for machine_id in MACHINE_CONFIGS:
        latest = (
            db.query(SensorReading.machine_phase, SensorReading.timestamp)
            .filter(SensorReading.machine_id == machine_id)
            .order_by(SensorReading.timestamp.desc())
            .first()
        )
        if latest is None or latest[0] != "FAILED":
            continue
        latest_ts = latest[1]
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=UTC)
        if (now - latest_ts).total_seconds() > 120:
            continue  # stale data — machine is paused or offline
        if not _acquire_lock(r, machine_id):
            continue
        job = {
            "phase": "START",
            "machine_id": machine_id,
            "decision_id": None,
            "alarm_id": None,
            "scenario": "EMERGENCY",
        }
        try:
            r.zadd(SCHEDULE_KEY, {json.dumps(job): now.timestamp()})
            logger.warning(
                f"Run-to-failure detected: {machine_id} FAILED — emergency repair dispatched"
            )
            scheduled += 1
        except Exception as exc:
            logger.error(f"Emergency scheduling failed for {machine_id}: {exc}")
            _release_lock(r, machine_id)
    return scheduled


def execute_due_jobs(db, r, now: datetime | None = None) -> int:
    """Run every schedule entry whose time has come. Returns jobs executed."""
    now_ts = (now or datetime.now(UTC)).timestamp()
    try:
        due = r.zrangebyscore(SCHEDULE_KEY, "-inf", now_ts)
    except Exception as exc:
        logger.error(f"Maintenance schedule read failed: {exc}")
        return 0

    executed = 0
    for raw in due:
        try:
            r.zrem(SCHEDULE_KEY, raw)
            job = json.loads(raw)
            if job.get("phase") == "START":
                _start_maintenance(db, r, job, now_ts)
            elif job.get("phase") == "COMPLETE":
                _complete_maintenance(db, r, job)
            executed += 1
        except Exception as exc:
            logger.error(f"Maintenance job failed ({raw!r}): {exc}")
            db.rollback()
    return executed


def _start_maintenance(db, r, job: dict, now_ts: float) -> None:
    from src.database.models import AlarmState, AlarmStateTransition, WorkOrder

    machine_id = job["machine_id"]
    line = machine_line(machine_id)
    publish_control(r, "PAUSE_LINE", line=line)

    now = datetime.now(UTC)
    alarm = None
    if job.get("alarm_id") is not None:
        alarm = db.query(AlarmState).filter(AlarmState.id == job["alarm_id"]).first()
    if alarm is not None and alarm.status != "OUT_OF_SERVICE":
        db.add(
            AlarmStateTransition(
                alarm_id=alarm.id,
                from_state=alarm.status,
                to_state="OUT_OF_SERVICE",
                operator_role="SYSTEM",
                reason=f"Maintenance started ({job['scenario']}); line {line} stopped",
                timestamp=now,
            )
        )
        alarm.status = "OUT_OF_SERVICE"
        alarm.oos_at = now

    work_order = None
    if job.get("decision_id") is not None:
        work_order = (
            db.query(WorkOrder).filter(WorkOrder.decision_id == job["decision_id"]).first()
        )
    if work_order is not None:
        work_order.status = "IN_PROGRESS"
    db.commit()

    duration = (
        EMERGENCY_SIM_HOURS if job["scenario"] == "EMERGENCY" else MAINTENANCE_SIM_HOURS
    )
    completion = dict(job, phase="COMPLETE", started_ts=now_ts)
    r.zadd(
        SCHEDULE_KEY,
        {
            json.dumps(completion): now_ts
            + _sim_hours_to_real_seconds(
                duration, cap_s=float(settings.MAINT_MAX_REAL_DURATION_S)
            )
        },
    )
    logger.info(
        f"Maintenance started: {machine_id} ({job['scenario']}) — line {line} paused "
        f"for {duration} sim-hours"
    )


def _complete_maintenance(db, r, job: dict) -> None:
    from src.database.models import (
        AlarmState,
        AlarmStateTransition,
        AnomalyLog,
        DecisionLog,
        MaintenanceLog,
        WorkOrder,
    )

    machine_id = job["machine_id"]
    line = machine_line(machine_id)
    now = datetime.now(UTC)

    publish_control(r, "RESET_MACHINE", machine_id=machine_id)
    publish_control(r, "RESUME_LINE", line=line)
    publish_control(r, "MAINTENANCE_DONE", machine_id=machine_id)

    decision = None
    if job.get("decision_id") is not None:
        decision = (
            db.query(DecisionLog).filter(DecisionLog.id == job["decision_id"]).first()
        )
    if job["scenario"] == "EMERGENCY":
        # The counterfactual happened: book the full reactive cost, no savings.
        from src.decision.machine_profiles import build_engine, build_profile, load_financials

        financials = load_financials(db)
        actual_cost = build_engine(machine_id, financials).run_to_failure_cost(
            build_profile(machine_id, financials)
        )
        avoided = None
    else:
        actual_cost = _planned_cost(decision) if decision is not None else 0.0
        context = _decision_context(decision) if decision is not None else {}
        avoided = context.get("run_to_failure_cost")

    # Close the anomaly trail for this machine.
    db.query(AnomalyLog).filter(
        AnomalyLog.machine_id == machine_id, AnomalyLog.status == "ACTIVE"
    ).update(
        {
            "status": "RESOLVED",
            "resolved_at": now,
            "resolution_type": job["scenario"],
        },
        synchronize_session=False,
    )

    alarm = None
    if job.get("alarm_id") is not None:
        alarm = db.query(AlarmState).filter(AlarmState.id == job["alarm_id"]).first()
    if alarm is not None:
        db.add(
            AlarmStateTransition(
                alarm_id=alarm.id,
                from_state=alarm.status,
                to_state="NORMAL",
                operator_role="SYSTEM",
                reason=f"Maintenance completed ({job['scenario']}); machine restored",
                timestamp=now,
            )
        )
        alarm.status = "NORMAL"
        alarm.oos_restored_at = now

    work_order = None
    if job.get("decision_id") is not None:
        work_order = (
            db.query(WorkOrder).filter(WorkOrder.decision_id == job["decision_id"]).first()
        )
    if work_order is not None:
        work_order.status = "COMPLETED"

    db.add(
        MaintenanceLog(
            machine_id=machine_id,
            alarm_id=job.get("alarm_id"),
            scheduled_at=datetime.fromtimestamp(job.get("started_ts", now.timestamp()), tz=UTC),
            performed_at=now,
            technician_notes=(
                "Reactive run-to-failure repair: crew called out after failure, "
                "fault diagnosed on a dead machine, parts fetched from warehouse. "
                "This is the cost the PdM system exists to avoid."
                if job["scenario"] == "EMERGENCY"
                else f"{job['scenario']} maintenance completed; machine restored to "
                f"full health. Avoided run-to-failure cost: "
                f"{avoided if avoided is not None else 'n/a'} EUR."
            ),
            fault_found=True,
            fault_description=work_order.fault_type if work_order is not None else None,
            downtime_minutes=int(
                (EMERGENCY_SIM_HOURS if job["scenario"] == "EMERGENCY" else MAINTENANCE_SIM_HOURS) * 60
                - (0 if job["scenario"] == "EMERGENCY" else SHIFT_CHANGE_OVERLAP_HOURS * 60)
            ),
            cost_eur=actual_cost,
        )
    )

    if decision is not None:
        decision.outcome = "MAINTENANCE_COMPLETED"
        decision.downtime_minutes = int(MAINTENANCE_SIM_HOURS * 60)

    db.commit()
    _release_lock(r, machine_id)
    logger.info(
        f"Maintenance completed: {machine_id} restored; line {line} resumed "
        f"(actual {actual_cost:.0f} EUR vs avoided {avoided} EUR)"
    )
