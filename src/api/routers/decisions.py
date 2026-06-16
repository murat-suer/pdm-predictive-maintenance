"""Decision queue endpoints: pending decisions and operator resolution."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_redis
from src.api.routers.common import ACTIVE_ALARM_STATUSES
from src.api.schemas import (
    DecisionResolveRequest,
    DecisionResolveResponse,
    DecisionScenarioItem,
    PendingDecision,
)
from src.database.models import AlarmState, AnomalyLog, DecisionLog, MachineHealthScore
from src.decision.decision_resolution import VALID_SCENARIOS, resolve_decision_plan

router = APIRouter(prefix="/decisions", tags=["decisions"])

# ---------------------------------------------------------------------------
# Human RBAC — scenario → minimum required role (ascending authority order).
# Bot resolutions (operator_id starts with "BOT-") bypass this check entirely
# so the auto-approval loop never regresses.
# ---------------------------------------------------------------------------
_ROLE_RANK: dict[str, int] = {
    "SUPERVISOR": 1,
    "PRODUCTION_MANAGER": 2,
    "PLANT_MANAGER": 3,
}

_SCENARIO_MIN_ROLE: dict[str, str] = {
    "OBSERVE": "SUPERVISOR",
    "DISPATCH_TECHNICIAN": "SUPERVISOR",
    "PLANNED": "SUPERVISOR",
    "REDUCE_LOAD": "PRODUCTION_MANAGER",
    "SHUTDOWN": "PLANT_MANAGER",
}


def _human_role_rank(role: str) -> int:
    """Return numeric rank for a human operator role; unknown roles get rank 0 (blocked)."""
    return _ROLE_RANK.get(role.upper(), 0)


def _check_human_rbac(operator_id: str | None, operator_role: str, scenario_id: str) -> None:
    """Raise HTTP 403 if a human operator lacks the authority for the requested scenario.

    Bot identities (operator_id starting with "BOT-") are exempt — their path
    is unchanged so the closed-loop auto-approval continues to work.
    """
    if operator_id and operator_id.upper().startswith("BOT-"):
        return  # bots bypass human RBAC

    min_role = _SCENARIO_MIN_ROLE.get(scenario_id.upper(), "PLANT_MANAGER")
    required_rank = _ROLE_RANK.get(min_role, 3)
    supplied_rank = _human_role_rank(operator_role)

    if supplied_rank < required_rank:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Insufficient authority: scenario '{scenario_id}' requires"
                f" '{min_role}' (rank {required_rank}),"
                f" but operator role '{operator_role}' has rank {supplied_rank}."
            ),
        )


def _to_pending(db: Session, decision: DecisionLog) -> PendingDecision:
    anomaly: AnomalyLog | None = None
    if decision.alarm_id is not None:
        alarm = db.query(AlarmState).filter(AlarmState.id == decision.alarm_id).first()
        if alarm is not None:
            anomaly = (
                db.query(AnomalyLog).filter(AnomalyLog.id == alarm.anomaly_id).first()
            )

    score = (
        db.query(MachineHealthScore)
        .filter(MachineHealthScore.machine_id == decision.machine_id)
        .order_by(MachineHealthScore.calculated_at.desc())
        .first()
    )

    scenarios = [
        DecisionScenarioItem(
            scenario=s.get("scenario", "UNKNOWN"),
            cost=float(s.get("cost", 0.0)),
            expected_cost=float(s.get("expected_cost", 0.0)),
            failure_probability=float(s.get("failure_probability", 0.0)),
            is_recommended=bool(s.get("is_recommended", False)),
        )
        for s in (decision.scenarios_presented or [])
    ]

    return PendingDecision(
        id=decision.id,
        machine_id=decision.machine_id,
        alarm_id=decision.alarm_id,
        severity=anomaly.severity if anomaly else None,
        fault_type=anomaly.fault_type if anomaly else None,
        anomaly_score=anomaly.anomaly_score if anomaly else None,
        shap_values=anomaly.shap_values if anomaly else None,
        rul_hours=score.rul_hours if score else None,
        ai_recommendation=decision.ai_recommendation,
        scenarios=scenarios,
        created_at=decision.created_at,
        due_at=decision.due_at,
    )


@router.get("/pending", response_model=list[PendingDecision])
async def pending_decisions(
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Decisions awaiting operator action, newest first."""
    decisions = (
        db.query(DecisionLog)
        .filter(DecisionLog.action == "PENDING")
        .order_by(DecisionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_to_pending(db, d) for d in decisions]


@router.post("/{decision_id}/resolve", response_model=DecisionResolveResponse)
async def resolve_decision(
    decision_id: str,
    request: DecisionResolveRequest,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Apply an operator-selected scenario to a pending decision."""
    scenario_id = request.scenario_id.upper()
    if scenario_id not in VALID_SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid scenario '{request.scenario_id}'. Valid: {sorted(VALID_SCENARIOS)}",
        )

    # Enforce role-based access for human operators (bots are exempt).
    _check_human_rbac(request.operator_id, request.operator_role, scenario_id)

    decision = db.query(DecisionLog).filter(DecisionLog.id == decision_id).first()
    if decision is None:
        raise HTTPException(status_code=404, detail=f"Decision not found: {decision_id}")
    if decision.action != "PENDING":
        raise HTTPException(
            status_code=409, detail=f"Decision already resolved: {decision.action}"
        )

    alarm = None
    if decision.alarm_id is not None:
        alarm = db.query(AlarmState).filter(AlarmState.id == decision.alarm_id).first()
    if alarm is None:
        alarm = (
            db.query(AlarmState)
            .filter(
                AlarmState.machine_id == decision.machine_id,
                AlarmState.status.in_(ACTIVE_ALARM_STATUSES),
            )
            .order_by(AlarmState.created_at.desc())
            .first()
        )
    if alarm is None:
        raise HTTPException(
            status_code=409,
            detail="No active alarm linked to this decision; cannot resolve.",
        )

    try:
        resolve_decision_plan(
            db=db,
            alarm=alarm,
            decision=decision,
            selected_scenario_id=scenario_id,
            operator_role=request.operator_role,
            operator_id=request.operator_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Decision resolution failed: {str(e)}") from e

    # A human-approved maintenance scenario enters the same lifecycle as a
    # bot-approved one: line stop → repair → full-health restart.
    try:
        from src.decision.maintenance import close_observation_cycle, schedule_from_decision

        close_observation_cycle(db, decision, alarm)
        schedule_from_decision(db, redis_client, decision, alarm)
    except Exception as e:
        # Scheduling failure must not fail the resolution itself.
        import logging

        logging.getLogger(__name__).error(f"Maintenance scheduling failed: {e}")

    return DecisionResolveResponse(
        id=decision.id,
        action=decision.action,
        chosen_scenario_id=decision.chosen_scenario_id,
        overridden=bool(decision.overridden),
        alarm_status=alarm.status,
        work_order_id=alarm.work_order_id,
    )
