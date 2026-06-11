"""Demo-time human-operator simulator.

EU AI Act framing: this is NOT an autonomous decision feature. In
production a human supervises every decision; during the unattended demo
that human is simulated. Each shift staffs one bot per authority rank,
every bot has its own operator ID, and the audit trail records exactly
who (human or which bot) made each call.

Authority limits:
  OPERATOR    may approve actions costing  < 500 EUR
  SUPERVISOR  may approve actions costing  < 2000 EUR
  MANAGER     approves everything above

The bot only ever acts after the human response window (due_at,
3 min ± 20%) has expired, and always selects the AI-recommended scenario.
"""
import logging
from datetime import UTC, datetime, time

from src.decision.decision_resolution import resolve_decision_plan

logger = logging.getLogger(__name__)

HUMAN_OPERATOR_ID = "HUMAN-OP-1"

OPERATOR_COST_LIMIT = 500.0
SUPERVISOR_COST_LIMIT = 2000.0

SHIFTS = (
    ("ALPHA", time(6, 0), time(14, 0)),
    ("BRAVO", time(14, 0), time(22, 0)),
    ("CHARLIE", time(22, 0), time(6, 0)),
)


def current_shift(now: datetime | None = None) -> str:
    """Shift name for a UTC timestamp (charlie wraps midnight)."""
    now = now or datetime.now(UTC)
    t = now.time()
    for name, start, end in SHIFTS:
        if start < end:
            if start <= t < end:
                return name
        elif t >= start or t < end:
            return name
    return "ALPHA"


def required_rank(action_cost: float) -> str:
    if action_cost < OPERATOR_COST_LIMIT:
        return "OPERATOR"
    if action_cost < SUPERVISOR_COST_LIMIT:
        return "SUPERVISOR"
    return "MANAGER"


def bot_identity(action_cost: float, now: datetime | None = None) -> tuple[str, str]:
    """(operator_id, operator_role) of the bot authorized for this cost."""
    rank = required_rank(action_cost)
    shift = current_shift(now)
    short = {"OPERATOR": "OPR", "SUPERVISOR": "SUP", "MANAGER": "MGR"}[rank]
    return f"BOT-{short}-{shift}", rank


def _scenario_cost(decision, scenario_id: str) -> float:
    for item in decision.scenarios_presented or []:
        if item.get("scenario") == scenario_id:
            return float(item.get("cost", 0.0))
    return 0.0


def act_on_due_decisions(db, redis_client=None, now: datetime | None = None) -> list:
    """Resolve every PENDING decision whose human window has expired.

    The bot selects the AI-recommended scenario through the full
    resolution chain (alarm transition + audit log), then hands the
    chosen scenario to the maintenance planner.
    """
    from src.database.models import AlarmState, DecisionLog

    now = now or datetime.now(UTC)
    resolved = []

    pending = (
        db.query(DecisionLog)
        .filter(DecisionLog.action == "PENDING", DecisionLog.due_at <= now)
        .all()
    )
    for decision in pending:
        scenario_id = decision.ai_recommendation
        if not scenario_id:
            # Scenarios were never attached (early fallback row) — observe.
            scenario_id = "OBSERVE"

        alarm = None
        if decision.alarm_id is not None:
            alarm = db.query(AlarmState).filter(AlarmState.id == decision.alarm_id).first()
        if alarm is None:
            logger.warning(
                f"Bot skip: decision {decision.id} has no alarm to act on"
            )
            continue

        operator_id, operator_role = bot_identity(
            _scenario_cost(decision, scenario_id), now
        )
        try:
            resolve_decision_plan(
                db=db,
                alarm=alarm,
                decision=decision,
                selected_scenario_id=scenario_id,
                operator_role=operator_role,
                operator_id=operator_id,
                auto_approve=True,
            )
            logger.info(
                f"Bot decision: {operator_id} approved {scenario_id} "
                f"for {decision.machine_id} (decision {decision.id})"
            )
            resolved.append(decision)
            from src.decision.maintenance import close_observation_cycle

            close_observation_cycle(db, decision, alarm)
            if redis_client is not None:
                from src.decision.maintenance import schedule_from_decision

                schedule_from_decision(db, redis_client, decision, alarm)
        except Exception as exc:
            logger.error(f"Bot resolution failed for {decision.id}: {exc}")
            db.rollback()

    return resolved
