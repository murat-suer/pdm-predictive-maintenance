"""
Decision Resolution (Phase 2D - Pipeline & Integration Layer).

Helpers for resolving a selected decision scenario:
- resolve_decision_plan(): Apply selected scenario to alarm + decision
- Alarm state transitions (SHELVED, OUT_OF_SERVICE)
- DecisionAuditLog creation
- Override detection (AI recommendation vs human choice)
- Response time calculation
- Work order ID generation (WO-{machine_id}-{HHMMSS})
"""

from datetime import UTC, datetime

from src.database.models import (
    AlarmStateTransition,
    DecisionAuditLog,
)

# ---------------------------------------------------------------------------
# Valid scenario IDs
# ---------------------------------------------------------------------------
VALID_SCENARIOS = {"OBSERVE", "DISPATCH_TECHNICIAN", "REDUCE_LOAD", "PLANNED", "SHUTDOWN"}

# Scenarios that transition alarm to SHELVED
SHELVE_SCENARIOS = {"PLANNED"}

# Scenarios that transition alarm to OUT_OF_SERVICE
OOS_SCENARIOS = {"SHUTDOWN"}

# Scenarios that keep alarm active (ACKNOWLEDGED)
ACTIVE_SCENARIOS = {"OBSERVE", "DISPATCH_TECHNICIAN", "REDUCE_LOAD"}


# ---------------------------------------------------------------------------
# Helper: generate_work_order_id
# ---------------------------------------------------------------------------
def generate_work_order_id(machine_id: str) -> str:
    """
    Generate a work order ID in format WO-{machine_id}-{HHMMSS}.

    Deterministic per second: same machine + same second → same ID.
    """
    now = datetime.now(UTC)
    time_part = now.strftime("%H%M%S")
    return f"WO-{machine_id}-{time_part}"


# ---------------------------------------------------------------------------
# Helper: calculate_response_time
# ---------------------------------------------------------------------------
def calculate_response_time(
    created_at: datetime, decided_at: datetime
) -> int:
    """
    Calculate response time in seconds between creation and decision.

    Returns 0 for negative durations (clock skew).
    Naive timestamps (e.g. from SQLite) are treated as UTC.
    """
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if decided_at.tzinfo is None:
        decided_at = decided_at.replace(tzinfo=UTC)
    delta = (decided_at - created_at).total_seconds()
    return max(0, int(delta))


# ---------------------------------------------------------------------------
# Helper: is_override
# ---------------------------------------------------------------------------
def is_override(decision) -> bool:
    """
    Check if a decision represents an override of the AI recommendation.

    Returns True if the chosen scenario differs from the AI recommendation.
    """
    if decision.overridden is True:
        return True
    if (
        decision.ai_recommendation
        and decision.chosen_scenario_id
        and decision.chosen_scenario_id != decision.ai_recommendation
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Main: resolve_decision_plan
# ---------------------------------------------------------------------------
def resolve_decision_plan(
    db,
    alarm,
    decision,
    selected_scenario_id: str,
    operator_role: str | None = None,
    operator_id: str | None = None,
    auto_approve: bool = False,
):
    """
    Resolve a decision by applying the selected scenario.

    Steps:
    1. Validate inputs
    2. Check if already resolved (idempotent)
    3. Detect override (AI recommendation vs human choice)
    4. Calculate response time
    5. Transition alarm state based on scenario
    6. Generate work order ID for SHUTDOWN
    7. Create DecisionAuditLog
    8. Commit to database

    Args:
        db: SQLAlchemy session
        alarm: AlarmState DB model instance
        decision: DecisionLog DB model instance
        selected_scenario_id: Which scenario was selected
        operator_role: Role of the operator (OPERATOR, SUPERVISOR, MANAGER)
        operator_id: ID of the operator
        auto_approve: If True, this is an auto-approval (no human)

    Returns:
        The decision object (for chaining)

    Raises:
        ValueError: If alarm or decision is None, or scenario is invalid
    """
    # Validate inputs
    if alarm is None:
        raise ValueError("alarm cannot be None")
    if decision is None:
        raise ValueError("decision cannot be None")
    if selected_scenario_id not in VALID_SCENARIOS:
        raise ValueError(
            f"Invalid scenario_id '{selected_scenario_id}'. "
            f"Must be one of: {VALID_SCENARIOS}"
        )

    # Check if already resolved (idempotent)
    if decision.action == "APPROVE" and decision.chosen_scenario_id is not None:
        return decision

    now = datetime.now(UTC)

    # Detect override
    overridden = selected_scenario_id != decision.ai_recommendation

    # Calculate response time
    response_time = calculate_response_time(decision.created_at, now)

    # Update decision fields
    decision.action = "APPROVE"
    decision.chosen_scenario_id = selected_scenario_id
    decision.decided_at = now
    decision.decided_by = operator_id or ("AUTO" if auto_approve else None)
    decision.operator_role = operator_role
    decision.response_time_s = response_time
    decision.overridden = overridden

    if auto_approve:
        decision.resolution_source = "AUTO"
        decision.auto_approved = True
    else:
        decision.resolution_source = "HUMAN"

    # Transition alarm state based on scenario
    old_status = alarm.status

    if selected_scenario_id in OOS_SCENARIOS:
        # SHUTDOWN → OUT_OF_SERVICE
        alarm.status = "OUT_OF_SERVICE"
        alarm.oos_at = now
        alarm.work_order_id = generate_work_order_id(alarm.machine_id)
    elif selected_scenario_id in SHELVE_SCENARIOS:
        # PLANNED → SHELVED
        alarm.status = "SHELVED"
        alarm.shelved_at = now
    elif selected_scenario_id in ACTIVE_SCENARIOS:
        # OBSERVE / REDUCE_LOAD → ACKNOWLEDGED (keep active)
        alarm.status = "ACKNOWLEDGED"

    # Create alarm state transition record
    transition = AlarmStateTransition(
        alarm_id=alarm.id,
        from_state=old_status,
        to_state=alarm.status,
        operator_role=operator_role,
        reason=f"Decision resolved: {selected_scenario_id}",
        timestamp=now,
    )
    db.add(transition)

    # Create DecisionAuditLog
    audit_log = DecisionAuditLog(
        decision_id=decision.id,
        alarm_id=alarm.id,
        action="APPROVE",
        scenario_id=selected_scenario_id,
        operator_role=operator_role,
        response_time_s=response_time,
        ai_recommendation=decision.ai_recommendation,
        overridden=overridden,
        override_reason=(
            f"Operator chose {selected_scenario_id} instead of "
            f"{decision.ai_recommendation}"
            if overridden
            else None
        ),
        escalation_level=decision.escalation_level or 1,
        created_at=now,
    )
    db.add(audit_log)

    # Commit
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    return decision
