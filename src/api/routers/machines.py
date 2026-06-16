"""Machine listing and detail endpoints backed by the database."""
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_redis
from src.api.routers.common import (
    _RECOMMENDATION_TO_STATUS,
    _last_repair_time,
    derive_machine_status,
    health_history,
    latest_health_scores,
    machine_type_label,
    utc_now,
)
from src.api.schemas import MachineDetail, MachineSummary, MachineTimeline, SensorSnapshot, TimelineEvent
from src.data_generator.machines import MACHINE_CONFIGS
from src.database.models import AnomalyLog, SensorReading

router = APIRouter(prefix="/machines", tags=["machines"])

SENSOR_HISTORY_POINTS = 30


@router.get("", response_model=list[MachineSummary])
async def list_machines(db: Session = Depends(get_db)):
    """List all configured machines with live status and health."""
    machine_ids = list(MACHINE_CONFIGS.keys())
    scores = latest_health_scores(db, machine_ids)

    summaries: list[MachineSummary] = []
    for machine_id in machine_ids:
        config = MACHINE_CONFIGS[machine_id]
        status, top_alarm = derive_machine_status(db, machine_id)
        score = scores.get(machine_id)
        summaries.append(
            MachineSummary(
                id=machine_id,
                name=config["name"],
                type=machine_type_label(machine_id),
                line=config["line"],
                status=status,
                health_score=score.health_score if score else None,
                rul_hours=score.rul_hours if score else None,
                reliability=round(score.reliability_score * 100.0, 1)
                if score and score.reliability_score is not None
                else None,
                classification=score.classification if score else None,
                top_alarm=top_alarm,
                health_history=health_history(db, machine_id),
            )
        )
    return summaries


@router.get("/{machine_id}", response_model=MachineDetail)
async def get_machine(machine_id: str, db: Session = Depends(get_db)):
    """Full machine detail: health, sensor snapshots and active faults."""
    config = MACHINE_CONFIGS.get(machine_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Unknown machine: {machine_id}")

    status, _ = derive_machine_status(db, machine_id)
    score = latest_health_scores(db, [machine_id]).get(machine_id)

    sensors: list[SensorSnapshot] = []
    for sensor_name, sensor_cfg in config["sensors"].items():
        rows = (
            db.query(SensorReading)
            .filter(
                SensorReading.machine_id == machine_id,
                SensorReading.sensor_name == sensor_name,
            )
            .order_by(SensorReading.timestamp.desc())
            .limit(SENSOR_HISTORY_POINTS)
            .all()
        )
        latest = rows[0] if rows else None
        sensors.append(
            SensorSnapshot(
                sensor_name=sensor_name,
                unit=sensor_cfg.get("unit"),
                value=latest.value if latest else None,
                timestamp=latest.timestamp if latest else None,
                warning_threshold=sensor_cfg.get("warning_threshold"),
                critical_threshold=sensor_cfg.get("critical_threshold"),
                nominal_mu=sensor_cfg.get("nominal_mu"),
                nominal_sigma=sensor_cfg.get("nominal_sigma"),
                degradation_direction=sensor_cfg.get("degradation_direction"),
                is_anomaly=bool(latest.is_anomaly) if latest else False,
                history=[r.value for r in reversed(rows)],
            )
        )

    active_faults = [
        {
            "fault_type": row.fault_type,
            "confidence": row.fault_confidence,
            "severity": row.severity,
            "top_contributing_sensor": row.top_contributing_sensor,
            "anomaly_score": row.anomaly_score,
            "detected_at": row.detected_at.isoformat(),
        }
        for row in (
            db.query(AnomalyLog)
            .filter(AnomalyLog.machine_id == machine_id, AnomalyLog.status == "ACTIVE")
            .order_by(AnomalyLog.detected_at.desc())
            .limit(5)
            .all()
        )
    ]

    return MachineDetail(
        id=machine_id,
        name=config["name"],
        type=machine_type_label(machine_id),
        line=config["line"],
        status=status,
        standard=config.get("standard"),
        failure_mode=config.get("failure_mode"),
        health_score=score.health_score if score else None,
        rul_hours=score.rul_hours if score else None,
        reliability=round(score.reliability_score * 100.0, 1)
        if score and score.reliability_score is not None
        else None,
        availability=score.availability_score if score else None,
        condition=score.condition_score if score else None,
        classification=score.classification if score else None,
        confidence=score.confidence if score else None,
        sensors=sensors,
        active_faults=active_faults,
    )


@router.get("/{machine_id}/sensors")
async def get_sensor_series(
    machine_id: str,
    minutes: int = Query(default=30, ge=1, le=1440),
    db: Session = Depends(get_db),
):
    """Raw sensor time series for the last N minutes, grouped per sensor."""
    if machine_id not in MACHINE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown machine: {machine_id}")

    cutoff = utc_now() - timedelta(minutes=minutes)
    rows = (
        db.query(SensorReading)
        .filter(SensorReading.machine_id == machine_id, SensorReading.timestamp >= cutoff)
        .order_by(SensorReading.timestamp.asc())
        .all()
    )

    series: dict = {}
    for row in rows:
        series.setdefault(row.sensor_name, []).append(
            {
                "timestamp": row.timestamp.isoformat(),
                "value": row.value,
                "is_anomaly": row.is_anomaly,
            }
        )
    return {"machine_id": machine_id, "minutes": minutes, "series": series}


_TIMELINE_CAP = 15


@router.get("/{machine_id}/timeline", response_model=MachineTimeline)
async def get_machine_timeline(machine_id: str, db: Session = Depends(get_db)):
    """Decision history since the machine's last real repair, newest first (capped at 15).

    'Since last repair' uses the same boundary as the latched-status logic:
    the most recent MaintenanceLog row whose downtime_minutes > 0.
    """
    from src.database.models import DecisionLog

    if machine_id not in MACHINE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown machine: {machine_id}")

    repaired_at = _last_repair_time(db, machine_id)

    # Only APPROVED decisions belong on the history — a pending recommendation
    # is not a committed action (the operator may not pick it), so it must not
    # show as an event until it is actually decided. Use the CHOSEN scenario
    # (what was approved), and the decided_at time, not the alarm-creation time.
    q = (
        db.query(
            DecisionLog.decided_at,
            DecisionLog.created_at,
            DecisionLog.chosen_scenario_id,
            DecisionLog.ai_recommendation,
            DecisionLog.outcome,
            DecisionLog.decided_by,
        )
        .filter(
            DecisionLog.machine_id == machine_id,
            DecisionLog.action == "APPROVE",
        )
    )
    if repaired_at is not None:
        q = q.filter(DecisionLog.decided_at > repaired_at)

    rows = q.order_by(DecisionLog.decided_at.desc()).limit(_TIMELINE_CAP).all()

    from src.api.routers.common import as_utc

    events: list[TimelineEvent] = []
    for decided_at, created_at, chosen, ai_rec, outcome, decided_by in rows:
        rec = chosen or ai_rec
        tier = _RECOMMENDATION_TO_STATUS.get(rec, "watch") if rec else "normal"
        events.append(
            TimelineEvent(
                at=as_utc(decided_at or created_at),
                recommendation=rec,
                tier=tier,
                outcome=outcome,
                decided_by=decided_by,
            )
        )

    return MachineTimeline(
        machine_id=machine_id,
        repaired_at=repaired_at,
        count=len(events),
        events=events,
    )


class WhatIfResult(BaseModel):
    """Defer-maintenance cost simulation for a machine."""
    machine_id: str
    rul_hours: float
    defer_hours: int
    act_now_cost_eur: float
    run_to_failure_cost_eur: float
    failure_probability: float
    deferred_risk_eur: float
    expected_deferred_cost_eur: float
    net_benefit_of_acting_now_eur: float
    breakeven_hours: int | None


@router.get("/{machine_id}/whatif", response_model=WhatIfResult)
async def whatif_defer_maintenance(
    machine_id: str,
    defer_hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """What happens if maintenance is deferred by N hours?

    Uses the decision engine's lognormal survival model and run-to-failure
    counterfactual: the expected deferred cost is the failure risk priced
    against the reactive repair, versus the planned repair cost of acting
    now. Breakeven is the deferral at which the expected failure cost
    alone exceeds the planned repair.
    """
    from src.decision.decision_engine import (
        DEFAULT_PLANNED_DOWNTIME_HOURS,
        SHIFT_CHANGE_OVERLAP_HOURS,
    )
    from src.decision.machine_profiles import build_engine, build_profile, load_financials

    if machine_id not in MACHINE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown machine: {machine_id}")

    score = latest_health_scores(db, [machine_id]).get(machine_id)
    rul = score.rul_hours if score and score.rul_hours is not None else None
    if rul is None:
        raise HTTPException(
            status_code=409,
            detail="No RUL estimate available yet for this machine.",
        )

    financials = load_financials(db)
    profile = build_profile(machine_id, financials)
    engine = build_engine(machine_id, financials)

    rtf = engine.run_to_failure_cost(profile)
    billable_downtime = max(
        DEFAULT_PLANNED_DOWNTIME_HOURS - SHIFT_CHANGE_OVERLAP_HOURS, 0.0
    )
    act_now = (
        engine.planned_labor_cost
        + engine.planned_parts_cost
        + profile.production_rate_per_hour * billable_downtime
    )

    p_fail = engine.survival_model.failure_within(float(rul), float(defer_hours))
    deferred_risk = p_fail * rtf
    expected_deferred = deferred_risk + (1.0 - p_fail) * act_now

    breakeven: int | None = None
    for h in range(1, 169):
        if engine.survival_model.failure_within(float(rul), float(h)) * rtf > act_now:
            breakeven = h
            break

    return WhatIfResult(
        machine_id=machine_id,
        rul_hours=round(float(rul), 1),
        defer_hours=defer_hours,
        act_now_cost_eur=round(act_now, 2),
        run_to_failure_cost_eur=round(rtf, 2),
        failure_probability=round(p_fail, 4),
        deferred_risk_eur=round(deferred_risk, 2),
        expected_deferred_cost_eur=round(expected_deferred, 2),
        net_benefit_of_acting_now_eur=round(expected_deferred - act_now, 2),
        breakeven_hours=breakeven,
    )


class InjectAnomalyRequest(BaseModel):
    """Demo control: trigger a fault scenario on a machine."""
    scenario: str = "full_cascade"
    ramp_seconds: float = 10.0


@router.post("/{machine_id}/inject-anomaly")
async def inject_anomaly(
    machine_id: str,
    request: InjectAnomalyRequest,
    redis_client=Depends(get_redis),
):
    """Publish a fault-injection command to the simulation (demo control).

    The data generator applies the scenario as a smooth degradation ramp;
    the full chain (detection -> alarm -> decision -> maintenance) then
    runs exactly as it would for an organic fault.
    """
    from src.data_generator.control import publish_control
    from src.data_generator.machines import ANOMALY_SCENARIOS

    if machine_id not in MACHINE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown machine: {machine_id}")
    if request.scenario not in ANOMALY_SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scenario '{request.scenario}'. Valid: {sorted(ANOMALY_SCENARIOS)}",
        )

    publish_control(
        redis_client,
        "INJECT_ANOMALY",
        machine_id=machine_id,
        scenario=request.scenario,
        ramp_seconds=request.ramp_seconds,
    )
    return {"status": "injected", "machine_id": machine_id, "scenario": request.scenario}
