"""Escalation policy for repeated OBSERVE decisions (EEMUA 191).

Watching an unexplained anomaly once is a legitimate call. Watching the same
machine for the third or fourth time is an alarm-management anti-pattern:
the operator sees that something is wrong but never learns what. This module
escalates instead:

- from the second consecutive OBSERVE, a DISPATCH_TECHNICIAN scenario is
  offered (on-line inspection, small call-out cost, no line stop) and — when
  the fault is still unclassified — recommended over watching again;
- after three consecutive OBSERVEs the OBSERVE option is no longer presented
  at all.

A real intervention (REDUCE_LOAD / PLANNED / SHUTDOWN) resets the streak;
a technician dispatch does not — it is part of the same watch cycle.
"""
import logging

logger = logging.getLogger(__name__)

DISPATCH_SCENARIO = "DISPATCH_TECHNICIAN"
# Offer the inspection from the 2nd consecutive decision after an OBSERVE.
OBSERVE_DISPATCH_AFTER = 1
# After this many consecutive OBSERVEs the option disappears entirely.
OBSERVE_MAX_STREAK = 3
# Technician call-out for an on-line inspection (no production stop).
DISPATCH_COST_EUR = 150.0

UNIDENTIFIED_FAULTS = (None, "", "UNCLASSIFIED_ANOMALY", "UNKNOWN")


def observe_streak(db, machine_id: str) -> int:
    """Consecutive OBSERVE approvals for a machine, newest first.

    DISPATCH_TECHNICIAN entries are skipped (same watch cycle); any real
    intervention breaks the streak.
    """
    from src.database.models import DecisionLog

    rows = (
        db.query(DecisionLog.chosen_scenario_id)
        .filter(
            DecisionLog.machine_id == machine_id,
            DecisionLog.action == "APPROVE",
        )
        .order_by(DecisionLog.decided_at.desc())
        .limit(10)
        .all()
    )
    streak = 0
    for (chosen,) in rows:
        if chosen == "OBSERVE":
            streak += 1
        elif chosen == DISPATCH_SCENARIO:
            continue
        else:
            break
    return streak


def fault_is_identified(db, anomaly_id) -> bool:
    """Whether the ML fault classifier put a name on this anomaly."""
    from src.database.models import AnomalyLog

    if anomaly_id is None:
        return False
    row = (
        db.query(AnomalyLog.fault_type)
        .filter(AnomalyLog.id == anomaly_id)
        .first()
    )
    return bool(row) and row[0] not in UNIDENTIFIED_FAULTS


def apply_observe_escalation(
    scenarios: list[dict],
    recommendation: str | None,
    streak: int,
    identified: bool,
) -> tuple[list[dict], str | None]:
    """Rewrite the scenario list according to the observe-streak policy.

    ``scenarios`` entries are the dicts persisted to
    DecisionLog.scenarios_presented (scenario / cost / expected_cost /
    failure_probability / is_recommended).
    """
    scenarios = [dict(s) for s in scenarios]
    observe = next((s for s in scenarios if s.get("scenario") == "OBSERVE"), None)

    if streak >= OBSERVE_MAX_STREAK and observe is not None:
        scenarios = [s for s in scenarios if s.get("scenario") != "OBSERVE"]
        if recommendation == "OBSERVE":
            recommendation = None

    if streak >= OBSERVE_DISPATCH_AFTER:
        # The inspection carries the same residual failure risk as watching —
        # its value is diagnostic, which the cost model deliberately does not
        # monetize. The policy, not the euro column, drives the recommendation.
        risk_term = float(observe["expected_cost"]) - float(observe["cost"]) if observe else 0.0
        dispatch = {
            "scenario": DISPATCH_SCENARIO,
            "cost": DISPATCH_COST_EUR,
            "expected_cost": round(DISPATCH_COST_EUR + risk_term, 2),
            "failure_probability": float(observe["failure_probability"]) if observe else 0.0,
            "is_recommended": False,
        }
        insert_at = 1 if scenarios and scenarios[0].get("scenario") == "OBSERVE" else 0
        scenarios.insert(insert_at, dispatch)
        if recommendation is None or (not identified and recommendation == "OBSERVE"):
            recommendation = DISPATCH_SCENARIO

    if recommendation is None and scenarios:
        best = min(scenarios, key=lambda s: float(s.get("expected_cost", float("inf"))))
        recommendation = best.get("scenario")

    for s in scenarios:
        s["is_recommended"] = s.get("scenario") == recommendation

    return scenarios, recommendation
