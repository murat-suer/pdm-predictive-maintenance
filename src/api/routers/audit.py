"""Unified audit-trail endpoint merging decision and alarm-state events."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.routers.common import as_utc
from src.api.schemas import AuditEvent, AuditPage
from src.database.models import AlarmState, AlarmStateTransition, DecisionAuditLog

router = APIRouter(prefix="/audit", tags=["audit"])

CRITICAL_TRANSITIONS = {"UNACKNOWLEDGED", "OUT_OF_SERVICE"}
WARNING_TRANSITIONS = {"SHELVED", "SUPPRESSED"}


def _transition_severity(to_state: str) -> str:
    if to_state in CRITICAL_TRANSITIONS:
        return "critical"
    if to_state in WARNING_TRANSITIONS:
        return "warning"
    return "info"


def _decision_events(db: Session, limit: int) -> list[AuditEvent]:
    rows = (
        db.query(DecisionAuditLog)
        .order_by(DecisionAuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    events = []
    for row in rows:
        decided_by = row.decision.decided_by if row.decision is not None else None
        machine_id = None
        if row.alarm_id is not None:
            alarm = db.query(AlarmState.machine_id).filter(AlarmState.id == row.alarm_id).first()
            machine_id = alarm[0] if alarm else None
        details_parts = []
        if row.ai_recommendation:
            details_parts.append(f"AI recommendation: {row.ai_recommendation}")
        if row.overridden:
            details_parts.append("operator overrode AI recommendation")
        if row.response_time_s is not None:
            details_parts.append(f"response time: {row.response_time_s}s")
        events.append(
            AuditEvent(
                id=f"DA-{row.id}",
                timestamp=as_utc(row.created_at),
                category="decision",
                severity="warning" if row.overridden else "info",
                actor=decided_by or row.operator_role or "SYSTEM",
                action=row.action,
                target=f"{machine_id or 'unknown'} :: {row.scenario_id or '-'}",
                details="; ".join(details_parts) or None,
            )
        )
    return events


def _alarm_events(db: Session, limit: int) -> list[AuditEvent]:
    rows = (
        db.query(AlarmStateTransition, AlarmState.machine_id)
        .join(AlarmState, AlarmStateTransition.alarm_id == AlarmState.id)
        .order_by(AlarmStateTransition.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        AuditEvent(
            id=f"AT-{t.id}",
            timestamp=as_utc(t.timestamp),
            category="alarm",
            severity=_transition_severity(t.to_state),
            actor=t.operator_role or "SYSTEM",
            action=f"ALARM_{t.to_state}",
            target=f"{machine_id} :: alarm #{t.alarm_id}",
            details=t.reason or f"State transition {t.from_state} → {t.to_state}",
        )
        for t, machine_id in rows
    ]


@router.get("/events", response_model=AuditPage)
async def audit_events(
    category: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Merged audit trail (decision audits + ISA-18.2 alarm transitions)."""
    events = _decision_events(db, limit + offset) + _alarm_events(db, limit + offset)
    if category:
        events = [e for e in events if e.category == category]
    if severity:
        events = [e for e in events if e.severity == severity]
    events.sort(key=lambda e: e.timestamp, reverse=True)
    total = len(events)
    return AuditPage(events=events[offset : offset + limit], total=total)
