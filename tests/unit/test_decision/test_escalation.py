"""
Unit tests for src.decision.escalation

Tests EscalationPolicy: role-based escalation logic based on severity,
cost thresholds, and alarm kinds.
"""

from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Import targets - these will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.escalation import (
    KINDS_REQUIRING_MANAGER,
    EscalationPolicy,
    Role,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def policy():
    """Default escalation policy with standard thresholds."""
    return EscalationPolicy()


@pytest.fixture
def custom_policy():
    """Policy with custom cost threshold."""
    return EscalationPolicy(cost_threshold=5000.0)


def _make_scenario(
    severity: str = "WARNING",
    estimated_cost: float = 1000.0,
    kind: str = "VIBRATION",
    rul_hours: float = 48.0,
    machine_id: str = "AC-001",
):
    """Helper to build a scenario dict."""
    return {
        "severity": severity,
        "estimated_cost": estimated_cost,
        "kind": kind,
        "rul_hours": rul_hours,
        "machine_id": machine_id,
    }


@dataclass
class ScenarioDataclass:
    """Dataclass variant for dual-format support testing."""
    severity: str = "WARNING"
    estimated_cost: float = 1000.0
    kind: str = "VIBRATION"
    rul_hours: float = 48.0
    machine_id: str = "AC-001"


# ---------------------------------------------------------------------------
# TestEscalationPolicyInit
# ---------------------------------------------------------------------------
class TestEscalationPolicyInit:
    def test_default_init(self, policy):
        """Default policy should have sensible defaults."""
        assert policy.cost_threshold > 0
        assert hasattr(policy, "role_hierarchy")

    def test_custom_threshold(self, custom_policy):
        assert custom_policy.cost_threshold == 5000.0


# ---------------------------------------------------------------------------
# TestSeverityEscalation
# ---------------------------------------------------------------------------
class TestSeverityEscalation:
    """EMERGENCY severity should always escalate to emergency role."""

    def test_emergency_severity_returns_emergency_role(self, policy):
        scenario = _make_scenario(severity="EMERGENCY", estimated_cost=100.0)
        role = policy.determine_role(scenario)
        assert role == Role.EMERGENCY

    def test_critical_severity_with_high_cost(self, policy):
        scenario = _make_scenario(severity="CRITICAL", estimated_cost=50000.0)
        role = policy.determine_role(scenario)
        assert role == Role.EMERGENCY

    def test_warning_severity_low_cost(self, policy):
        scenario = _make_scenario(severity="WARNING", estimated_cost=100.0)
        role = policy.determine_role(scenario)
        assert role == Role.OPERATOR


# ---------------------------------------------------------------------------
# TestCostBasedEscalation
# ---------------------------------------------------------------------------
class TestCostBasedEscalation:
    """High estimated cost should trigger emergency escalation."""

    def test_cost_above_threshold_triggers_emergency(self, policy):
        scenario = _make_scenario(
            severity="WARNING",
            estimated_cost=policy.cost_threshold + 1.0,
        )
        role = policy.determine_role(scenario)
        assert role == Role.EMERGENCY

    def test_cost_at_threshold_triggers_emergency(self, policy):
        """At exactly the threshold, should still escalate."""
        scenario = _make_scenario(
            severity="WARNING",
            estimated_cost=policy.cost_threshold,
        )
        role = policy.determine_role(scenario)
        assert role == Role.EMERGENCY

    def test_cost_below_threshold_stays_operator(self, policy):
        scenario = _make_scenario(
            severity="WARNING",
            estimated_cost=policy.cost_threshold - 1.0,
        )
        role = policy.determine_role(scenario)
        assert role == Role.OPERATOR

    def test_zero_cost_is_operator(self, policy):
        scenario = _make_scenario(severity="WARNING", estimated_cost=0.0)
        role = policy.determine_role(scenario)
        assert role == Role.OPERATOR


# ---------------------------------------------------------------------------
# TestKindsRequiringManager
# ---------------------------------------------------------------------------
class TestKindsRequiringManager:
    """Certain alarm kinds should escalate to manager regardless of cost."""

    def test_each_kind_requiring_manager(self, policy):
        for kind in KINDS_REQUIRING_MANAGER:
            scenario = _make_scenario(
                severity="WARNING",
                estimated_cost=100.0,
                kind=kind,
            )
            role = policy.determine_role(scenario)
            assert role == Role.MANAGER, f"Kind {kind} should require MANAGER"

    def test_non_manager_kind_stays_operator(self, policy):
        scenario = _make_scenario(
            severity="WARNING",
            estimated_cost=100.0,
            kind="ROUTINE_INSPECTION",
        )
        role = policy.determine_role(scenario)
        assert role == Role.OPERATOR

    def test_kinds_requiring_manager_is_not_empty(self):
        assert len(KINDS_REQUIRING_MANAGER) > 0


# ---------------------------------------------------------------------------
# TestBatchRoleAssignment
# ---------------------------------------------------------------------------
class TestBatchRoleAssignment:
    """Batch role assignment for multiple scenarios."""

    def test_batch_assignment(self, policy):
        scenarios = [
            _make_scenario(severity="EMERGENCY", machine_id="M1"),
            _make_scenario(severity="WARNING", estimated_cost=100.0, machine_id="M2"),
            _make_scenario(severity="WARNING", estimated_cost=100.0, kind=KINDS_REQUIRING_MANAGER[0], machine_id="M3"),
        ]
        roles = policy.determine_roles(scenarios)
        assert len(roles) == 3
        assert roles[0] == Role.EMERGENCY
        assert roles[1] == Role.OPERATOR
        assert roles[2] == Role.MANAGER

    def test_batch_empty_list(self, policy):
        roles = policy.determine_roles([])
        assert roles == []


# ---------------------------------------------------------------------------
# TestDataclassAndDictSupport
# ---------------------------------------------------------------------------
class TestDataclassAndDictSupport:
    """Policy should accept both dict and dataclass scenarios."""

    def test_dict_scenario(self, policy):
        scenario = _make_scenario(severity="EMERGENCY")
        role = policy.determine_role(scenario)
        assert role == Role.EMERGENCY

    def test_dataclass_scenario(self, policy):
        scenario = ScenarioDataclass(severity="EMERGENCY")
        role = policy.determine_role(scenario)
        assert role == Role.EMERGENCY

    def test_dict_and_dataclass_same_result(self, policy):
        d = _make_scenario(severity="WARNING", estimated_cost=100.0)
        dc = ScenarioDataclass(severity="WARNING", estimated_cost=100.0)
        assert policy.determine_role(d) == policy.determine_role(dc)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_negative_cost(self, policy):
        """Negative cost should not crash; treat as 0 or low."""
        scenario = _make_scenario(severity="WARNING", estimated_cost=-100.0)
        role = policy.determine_role(scenario)
        # Negative cost should not trigger emergency
        assert role in (Role.OPERATOR, Role.SUPERVISOR)

    def test_unknown_severity(self, policy):
        """Unknown severity should default to safe (operator) escalation."""
        scenario = _make_scenario(severity="UNKNOWN_SEV", estimated_cost=100.0)
        role = policy.determine_role(scenario)
        assert role == Role.OPERATOR

    def test_very_large_cost(self, policy):
        scenario = _make_scenario(severity="WARNING", estimated_cost=1e12)
        role = policy.determine_role(scenario)
        assert role == Role.EMERGENCY

    def test_role_hierarchy_ordering(self, policy):
        """Operator < Supervisor < Manager < Emergency in hierarchy."""
        hierarchy = policy.role_hierarchy
        assert hierarchy.index(Role.OPERATOR) < hierarchy.index(Role.SUPERVISOR)
        assert hierarchy.index(Role.SUPERVISOR) < hierarchy.index(Role.MANAGER)
        assert hierarchy.index(Role.MANAGER) < hierarchy.index(Role.EMERGENCY)
