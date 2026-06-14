"""Analytics aggregation endpoints that power the charts dashboard.

All endpoints share the same windowing pattern from savings.py:
- Default window = since the earliest MachineHealthScore row (demo cycle start).
- Override with ?hours= to narrow the window.
- All responses include window_started_at (ISO) so the frontend knows the period.

SQLite portability note: date_trunc / time_bucket are Postgres-only.
Bucketing is therefore done in Python after fetching raw rows.
"""
import json
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.routers.common import as_utc, utc_now
from src.api.routers.savings import AVERTING_SCENARIOS
from src.database.models import (
    AnomalyLog,
    DecisionLog,
    MachineHealthScore,
    MaintenanceLog,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])

# ─── Shared window helper ─────────────────────────────────────────────────


def _resolve_window(db: Session, hours: int | None) -> tuple[Any, Any]:
    """Return (cutoff datetime, cycle_start raw value).

    cycle_start is the raw DB value of min(MachineHealthScore.calculated_at).
    """
    cycle_start = db.query(func.min(MachineHealthScore.calculated_at)).scalar()
    now = utc_now()
    if hours is not None:
        cutoff = now - timedelta(hours=hours)
    elif cycle_start is not None:
        cutoff = as_utc(cycle_start)
    else:
        cutoff = now - timedelta(hours=24)
    return cutoff, cycle_start


def _window_started_at(cycle_start: Any) -> str | None:
    if cycle_start is None:
        return None
    return as_utc(cycle_start).isoformat()


# ─── 1. Decision Mix ──────────────────────────────────────────────────────


class DecisionMixItem(BaseModel):
    scenario: str | None
    count: int


class DecisionMixResponse(BaseModel):
    window_started_at: str | None
    total: int
    items: list[DecisionMixItem]


@router.get("/decision-mix", response_model=DecisionMixResponse)
async def decision_mix(
    hours: int | None = Query(default=None, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
):
    """Counts of APPROVE decisions grouped by chosen_scenario_id."""
    cutoff, cycle_start = _resolve_window(db, hours)
    rows = (
        db.query(DecisionLog.chosen_scenario_id, func.count(DecisionLog.id))
        .filter(
            DecisionLog.action == "APPROVE",
            DecisionLog.created_at >= cutoff,
        )
        .group_by(DecisionLog.chosen_scenario_id)
        .all()
    )
    items = [DecisionMixItem(scenario=r[0], count=r[1]) for r in rows]
    total = sum(i.count for i in items)
    return DecisionMixResponse(
        window_started_at=_window_started_at(cycle_start),
        total=total,
        items=items,
    )


# ─── 2. Fault Distribution ───────────────────────────────────────────────


class FaultDistributionItem(BaseModel):
    fault_type: str | None
    count: int


class FaultDistributionResponse(BaseModel):
    window_started_at: str | None
    total: int
    items: list[FaultDistributionItem]


@router.get("/fault-distribution", response_model=FaultDistributionResponse)
async def fault_distribution(
    hours: int | None = Query(default=None, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
):
    """Counts of anomalies grouped by fault_type (includes UNCLASSIFIED_ANOMALY)."""
    cutoff, cycle_start = _resolve_window(db, hours)
    rows = (
        db.query(AnomalyLog.fault_type, func.count(AnomalyLog.id))
        .filter(AnomalyLog.detected_at >= cutoff)
        .group_by(AnomalyLog.fault_type)
        .all()
    )
    items = [
        FaultDistributionItem(
            fault_type=r[0] if r[0] is not None else "UNCLASSIFIED_ANOMALY",
            count=r[1],
        )
        for r in rows
    ]
    total = sum(i.count for i in items)
    return FaultDistributionResponse(
        window_started_at=_window_started_at(cycle_start),
        total=total,
        items=items,
    )


# ─── 3. Savings Timeseries ───────────────────────────────────────────────


class SavingsPoint(BaseModel):
    t: str
    cumulative_eur: float
    incremental_eur: float


class SavingsTimeseriesResponse(BaseModel):
    window_started_at: str | None
    points: list[SavingsPoint]


def _approve_decision_for_alarm(db: Session, alarm_id: int | None) -> DecisionLog | None:
    """Return the most recent APPROVE decision for an alarm, or None."""
    if alarm_id is None:
        return None
    return (
        db.query(DecisionLog)
        .filter(
            DecisionLog.alarm_id == alarm_id,
            DecisionLog.action == "APPROVE",
        )
        .order_by(DecisionLog.created_at.desc())
        .first()
    )


def _get_avoided_cost(decision: DecisionLog | None) -> float | None:
    """Extract run_to_failure_cost from decision notes (same logic as savings.py)."""
    if decision is None:
        return None
    if decision.chosen_scenario_id not in AVERTING_SCENARIOS:
        return None
    try:
        context = json.loads(decision.notes) if decision.notes else {}
        return float(context.get("run_to_failure_cost"))
    except (TypeError, ValueError):
        return None


@router.get("/savings-timeseries", response_model=SavingsTimeseriesResponse)
async def savings_timeseries(
    buckets: int = Query(default=12, ge=1, le=200),
    hours: int | None = Query(default=None, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
):
    """Cumulative avoided cost over time, bucketed across the window.

    Only completed maintenances whose decision chose an AVERTING_SCENARIOS
    scenario contribute. Bucket boundaries are evenly spaced across the window.
    """
    cutoff, cycle_start = _resolve_window(db, hours)
    now = utc_now()

    logs = (
        db.query(MaintenanceLog)
        .filter(
            MaintenanceLog.performed_at.isnot(None),
            MaintenanceLog.performed_at >= cutoff,
        )
        .order_by(MaintenanceLog.performed_at.asc())
        .all()
    )

    # Build (performed_at, savings_eur) pairs
    events: list[tuple[Any, float]] = []
    for log in logs:
        decision = _approve_decision_for_alarm(db, log.alarm_id)
        avoided = _get_avoided_cost(decision)
        if avoided is None:
            continue
        actual = float(log.cost_eur or 0.0)
        savings = avoided - actual
        events.append((as_utc(log.performed_at), savings))

    # Produce bucket boundaries
    window_start = as_utc(cycle_start) if cycle_start is not None else cutoff
    total_seconds = (now - window_start).total_seconds()
    if total_seconds <= 0:
        return SavingsTimeseriesResponse(
            window_started_at=_window_started_at(cycle_start),
            points=[],
        )

    bucket_width = total_seconds / buckets
    cumulative = 0.0
    points: list[SavingsPoint] = []

    for i in range(buckets):
        bucket_end = window_start + timedelta(seconds=bucket_width * (i + 1))
        incremental = sum(
            s for t, s in events if t <= bucket_end
        )
        # incremental per bucket = cumulative at bucket_end - previous cumulative
        bucket_incremental = incremental - cumulative
        cumulative = incremental
        points.append(
            SavingsPoint(
                t=bucket_end.isoformat(),
                cumulative_eur=round(cumulative, 2),
                incremental_eur=round(bucket_incremental, 2),
            )
        )

    return SavingsTimeseriesResponse(
        window_started_at=_window_started_at(cycle_start),
        points=points,
    )


# ─── 4. MHI History ──────────────────────────────────────────────────────


class MHIPoint(BaseModel):
    t: str
    mhi: float


class MachineHistory(BaseModel):
    machine_id: str
    points: list[MHIPoint]


class MHIHistoryResponse(BaseModel):
    window_started_at: str | None
    machines: list[MachineHistory]


@router.get("/mhi-history", response_model=MHIHistoryResponse)
async def mhi_history(
    buckets: int = Query(default=24, ge=1, le=500),
    hours: int | None = Query(default=None, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
):
    """Per-machine MHI (0-100) averaged per bucket over the window.

    health_score is stored 0-1 in the DB; multiply by 100 for the response.
    """
    cutoff, cycle_start = _resolve_window(db, hours)
    now = utc_now()

    rows = (
        db.query(
            MachineHealthScore.machine_id,
            MachineHealthScore.calculated_at,
            MachineHealthScore.health_score,
        )
        .filter(MachineHealthScore.calculated_at >= cutoff)
        .order_by(MachineHealthScore.calculated_at.asc())
        .all()
    )

    if not rows:
        return MHIHistoryResponse(
            window_started_at=_window_started_at(cycle_start),
            machines=[],
        )

    # Group rows by machine
    by_machine: dict[str, list[tuple[Any, float]]] = {}
    for machine_id, calculated_at, health_score in rows:
        by_machine.setdefault(machine_id, []).append(
            (as_utc(calculated_at), float(health_score) * 100.0)
        )

    window_start = as_utc(cycle_start) if cycle_start is not None else cutoff
    total_seconds = (now - window_start).total_seconds()
    if total_seconds <= 0:
        return MHIHistoryResponse(
            window_started_at=_window_started_at(cycle_start),
            machines=[],
        )

    bucket_width = total_seconds / buckets
    machines_out: list[MachineHistory] = []

    for machine_id, machine_rows in sorted(by_machine.items()):
        points: list[MHIPoint] = []
        for i in range(buckets):
            bucket_start = window_start + timedelta(seconds=bucket_width * i)
            bucket_end = window_start + timedelta(seconds=bucket_width * (i + 1))
            in_bucket = [mhi for t, mhi in machine_rows if bucket_start <= t < bucket_end]
            if not in_bucket:
                continue
            avg_mhi = sum(in_bucket) / len(in_bucket)
            points.append(
                MHIPoint(t=bucket_end.isoformat(), mhi=round(avg_mhi, 1))
            )
        if points:
            machines_out.append(MachineHistory(machine_id=machine_id, points=points))

    return MHIHistoryResponse(
        window_started_at=_window_started_at(cycle_start),
        machines=machines_out,
    )


# ─── 5. Maintenance Timeline ─────────────────────────────────────────────


class MaintenanceEvent(BaseModel):
    machine_id: str
    scenario: str | None
    performed_at: str
    actual_cost_eur: float
    savings_eur: float | None
    downtime_minutes: int | None


class MaintenanceTimelineResponse(BaseModel):
    window_started_at: str | None
    events: list[MaintenanceEvent]


@router.get("/maintenance-timeline", response_model=MaintenanceTimelineResponse)
async def maintenance_timeline(
    hours: int | None = Query(default=None, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
):
    """Recent completed maintenances ordered by performed_at desc, capped at 50."""
    cutoff, cycle_start = _resolve_window(db, hours)
    logs = (
        db.query(MaintenanceLog)
        .filter(
            MaintenanceLog.performed_at.isnot(None),
            MaintenanceLog.performed_at >= cutoff,
        )
        .order_by(MaintenanceLog.performed_at.desc())
        .limit(50)
        .all()
    )

    events: list[MaintenanceEvent] = []
    for log in logs:
        decision = _approve_decision_for_alarm(db, log.alarm_id)
        scenario = decision.chosen_scenario_id if decision else None
        avoided = _get_avoided_cost(decision)
        actual = float(log.cost_eur or 0.0)
        savings = round(avoided - actual, 2) if avoided is not None else None
        events.append(
            MaintenanceEvent(
                machine_id=log.machine_id,
                scenario=scenario,
                performed_at=as_utc(log.performed_at).isoformat(),
                actual_cost_eur=round(actual, 2),
                savings_eur=savings,
                downtime_minutes=log.downtime_minutes,
            )
        )

    return MaintenanceTimelineResponse(
        window_started_at=_window_started_at(cycle_start),
        events=events,
    )


# ─── 6. Decision Stats ───────────────────────────────────────────────────


class ActorCount(BaseModel):
    decided_by: str | None
    count: int


class BotVsHuman(BaseModel):
    bot: int
    human: int


class DecisionStatsResponse(BaseModel):
    window_started_at: str | None
    total: int
    auto_approved: int
    overridden: int
    override_rate: float
    avg_response_time_s: float | None
    by_actor: list[ActorCount]
    bot_vs_human: BotVsHuman


@router.get("/decision-stats", response_model=DecisionStatsResponse)
async def decision_stats(
    hours: int | None = Query(default=None, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
):
    """Aggregate decision statistics for APPROVE-action rows only."""
    cutoff, cycle_start = _resolve_window(db, hours)
    rows = (
        db.query(DecisionLog)
        .filter(
            DecisionLog.action == "APPROVE",
            DecisionLog.created_at >= cutoff,
        )
        .all()
    )

    total = len(rows)
    auto_approved = sum(1 for r in rows if r.auto_approved)
    overridden = sum(1 for r in rows if r.overridden)
    override_rate = round(overridden / total, 4) if total > 0 else 0.0

    response_times = [r.response_time_s for r in rows if r.response_time_s is not None]
    avg_response_time_s = (
        round(sum(response_times) / len(response_times), 2)
        if response_times
        else None
    )

    # Group by decided_by
    actor_counts: dict[str | None, int] = {}
    bot_count = 0
    human_count = 0
    for r in rows:
        actor_counts[r.decided_by] = actor_counts.get(r.decided_by, 0) + 1
        if r.decided_by is not None and r.decided_by.startswith("BOT-"):
            bot_count += 1
        else:
            human_count += 1

    by_actor = [
        ActorCount(decided_by=actor, count=cnt)
        for actor, cnt in sorted(actor_counts.items(), key=lambda x: -x[1])
    ]

    return DecisionStatsResponse(
        window_started_at=_window_started_at(cycle_start),
        total=total,
        auto_approved=auto_approved,
        overridden=overridden,
        override_rate=override_rate,
        avg_response_time_s=avg_response_time_s,
        by_actor=by_actor,
        bot_vs_human=BotVsHuman(bot=bot_count, human=human_count),
    )
