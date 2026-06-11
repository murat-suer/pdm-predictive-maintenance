"""Alarm listing endpoints (ISA-18.2 alarm states joined with anomalies)."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.routers.common import ACTIVE_ALARM_STATUSES, as_utc, utc_now
from src.api.schemas import AlarmItem
from src.database.models import AlarmState, AnomalyLog

router = APIRouter(prefix="/alarms", tags=["alarms"])


@router.get("", response_model=list[AlarmItem])
async def list_alarms(
    active: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List alarms, newest first. With active=true only unresolved states."""
    query = (
        db.query(AlarmState, AnomalyLog)
        .join(AnomalyLog, AlarmState.anomaly_id == AnomalyLog.id)
        .order_by(AlarmState.created_at.desc())
    )
    if active:
        query = query.filter(AlarmState.status.in_(ACTIVE_ALARM_STATUSES))

    now = utc_now()
    items: list[AlarmItem] = []
    for alarm, anomaly in query.limit(limit).all():
        created_at = as_utc(alarm.created_at)
        duration_minutes = max(0, int((now - created_at).total_seconds() // 60))
        items.append(
            AlarmItem(
                id=alarm.id,
                machine_id=alarm.machine_id,
                status=alarm.status,
                level=alarm.level,
                severity=anomaly.severity,
                fault_type=anomaly.fault_type,
                top_contributing_sensor=anomaly.top_contributing_sensor,
                anomaly_score=anomaly.anomaly_score,
                created_at=created_at,
                duration_minutes=duration_minutes,
            )
        )
    return items
