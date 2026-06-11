"""Savings accounting: what the PdM system saved vs running to failure.

Each completed maintenance is compared with its recorded counterfactual —
the run-to-failure cost captured on the decision at recommendation time
(emergency repair premium, extended reactive downtime, cascade losses,
collateral stress). Savings = avoided − actual.
"""
import json
from datetime import timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.routers.common import as_utc, utc_now
from src.database.models import DecisionLog, MaintenanceLog

router = APIRouter(prefix="/savings", tags=["savings"])

# Only an actual intervention averts the run-to-failure counterfactual.
# OBSERVE/DISPATCH_TECHNICIAN keep the risk open, so they book cost but
# never claim avoided money.
AVERTING_SCENARIOS = {"REDUCE_LOAD", "PLANNED", "SHUTDOWN"}


class SavingsEvent(BaseModel):
    machine_id: str
    scenario: str | None
    performed_at: str
    decided_by: str | None
    actual_cost_eur: float
    avoided_cost_eur: float | None
    savings_eur: float | None
    downtime_minutes: int | None


class SavingsSummary(BaseModel):
    events: list[SavingsEvent]
    total_actual_eur: float
    total_avoided_eur: float
    total_savings_eur: float
    maintenance_count: int
    window_hours: int


def _decision_for_alarm(db: Session, alarm_id) -> DecisionLog | None:
    if alarm_id is None:
        return None
    return (
        db.query(DecisionLog)
        .filter(DecisionLog.alarm_id == alarm_id)
        .order_by(DecisionLog.created_at.desc())
        .first()
    )


@router.get("", response_model=SavingsSummary)
async def savings_summary(
    hours: int = Query(default=24, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
):
    """Avoided-cost accounting across completed maintenances in a window."""
    cutoff = utc_now() - timedelta(hours=hours)
    logs = (
        db.query(MaintenanceLog)
        .filter(
            MaintenanceLog.performed_at.isnot(None),
            MaintenanceLog.performed_at >= cutoff,
        )
        .order_by(MaintenanceLog.performed_at.desc())
        .limit(200)
        .all()
    )

    events: list[SavingsEvent] = []
    total_actual = 0.0
    total_avoided = 0.0
    for log in logs:
        decision = _decision_for_alarm(db, log.alarm_id)
        avoided = None
        scenario = None
        decided_by = None
        if decision is not None:
            scenario = decision.chosen_scenario_id
            decided_by = decision.decided_by
            if scenario in AVERTING_SCENARIOS:
                try:
                    context = json.loads(decision.notes) if decision.notes else {}
                    avoided = float(context.get("run_to_failure_cost"))
                except (TypeError, ValueError):
                    avoided = None

        actual = float(log.cost_eur or 0.0)
        savings = (avoided - actual) if avoided is not None else None
        total_actual += actual
        if avoided is not None:
            total_avoided += avoided

        events.append(
            SavingsEvent(
                machine_id=log.machine_id,
                scenario=scenario,
                performed_at=as_utc(log.performed_at).isoformat(),
                decided_by=decided_by,
                actual_cost_eur=round(actual, 2),
                avoided_cost_eur=round(avoided, 2) if avoided is not None else None,
                savings_eur=round(savings, 2) if savings is not None else None,
                downtime_minutes=log.downtime_minutes,
            )
        )

    return SavingsSummary(
        events=events,
        total_actual_eur=round(total_actual, 2),
        total_avoided_eur=round(total_avoided, 2),
        total_savings_eur=round(total_avoided - total_actual, 2),
        maintenance_count=len(events),
        window_hours=hours,
    )
