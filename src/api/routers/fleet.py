"""Fleet-level KPI and trend endpoints."""
from datetime import timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.routers.common import (
    ACTIVE_ALARM_STATUSES,
    derive_machine_status,
    latest_health_scores,
    utc_now,
)
from src.api.schemas import FleetSummary, HealthTrendPoint
from src.data_generator.machines import MACHINE_CONFIGS
from src.database.models import AlarmState, MachineHealthScore

router = APIRouter(prefix="/fleet", tags=["fleet"])


@router.get("/summary", response_model=FleetSummary)
async def fleet_summary(db: Session = Depends(get_db)):
    """KPI counts for the navbar and fleet overview cards."""
    machine_ids = list(MACHINE_CONFIGS.keys())
    counts = {"normal": 0, "warning": 0, "critical": 0, "maintenance": 0, "offline": 0}
    for machine_id in machine_ids:
        status, _ = derive_machine_status(db, machine_id)
        counts[status] += 1

    scores = latest_health_scores(db, machine_ids)
    reliabilities = [
        s.reliability_score for s in scores.values() if s.reliability_score is not None
    ]
    avg_reliability = (
        round(sum(reliabilities) / len(reliabilities) * 100.0, 1) if reliabilities else None
    )

    active_alarms = (
        db.query(func.count(AlarmState.id))
        .filter(AlarmState.status.in_(ACTIVE_ALARM_STATUSES))
        .scalar()
        or 0
    )

    return FleetSummary(
        total=len(machine_ids),
        normal=counts["normal"],
        warning=counts["warning"],
        critical=counts["critical"],
        maintenance=counts["maintenance"],
        offline=counts["offline"],
        avg_reliability=avg_reliability,
        active_alarms=active_alarms,
    )


@router.get("/health-trend", response_model=list[HealthTrendPoint])
async def fleet_health_trend(
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """Fleet-average health score bucketed per hour, oldest first."""
    cutoff = utc_now() - timedelta(hours=hours)
    rows = (
        db.query(MachineHealthScore.calculated_at, MachineHealthScore.health_score)
        .filter(MachineHealthScore.calculated_at >= cutoff)
        .order_by(MachineHealthScore.calculated_at.asc())
        .all()
    )

    buckets: dict = {}
    for calculated_at, health_score in rows:
        bucket = calculated_at.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(bucket, []).append(health_score)

    return [
        HealthTrendPoint(bucket=bucket, avg_health_score=round(sum(vals) / len(vals), 4))
        for bucket, vals in sorted(buckets.items())
    ]
