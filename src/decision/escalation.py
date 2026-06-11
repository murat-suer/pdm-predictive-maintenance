"""
Role-based escalation policy for PDM v3.

Determines the required role (OPERATOR, SUPERVISOR, MANAGER, EMERGENCY)
based on severity, estimated cost, and alarm kind.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Escalation roles in ascending authority order."""
    OPERATOR = "OPERATOR"
    SUPERVISOR = "SUPERVISOR"
    MANAGER = "MANAGER"
    EMERGENCY = "EMERGENCY"


# Cost threshold above which escalation is EMERGENCY (default)
EMERGENCY_THRESHOLD = 10000.0

# Alarm kinds that always require MANAGER regardless of cost
KINDS_REQUIRING_MANAGER = (
    "SAFETY_CRITICAL",
    "CASCADE_FAILURE",
    "PRODUCTION_LINE_STOP",
    "ENVIRONMENTAL_BREACH",
)

# Severity → base role mapping
_SEVERITY_ROLE = {
    "EMERGENCY": Role.EMERGENCY,
    "CRITICAL": Role.MANAGER,
    "WARNING": Role.OPERATOR,
    "INFO": Role.OPERATOR,
}


@dataclass
class Scenario:
    """A decision scenario for escalation evaluation."""
    severity: str = "WARNING"
    estimated_cost: float = 0.0
    kind: str = "VIBRATION"
    rul_hours: float = 48.0
    machine_id: str = ""


class EscalationPolicy:
    """
    Determines the required role for a given scenario.

    Rules (in priority order):
    1. severity == EMERGENCY → EMERGENCY
    2. estimated_cost >= cost_threshold → EMERGENCY
    3. kind in KINDS_REQUIRING_MANAGER → MANAGER
    4. severity == CRITICAL and cost >= cost_threshold → EMERGENCY
    5. severity mapping → base role
    """

    def __init__(self, cost_threshold: float = EMERGENCY_THRESHOLD):
        self.cost_threshold = cost_threshold
        self.role_hierarchy: list[Role] = [
            Role.OPERATOR,
            Role.SUPERVISOR,
            Role.MANAGER,
            Role.EMERGENCY,
        ]

    def _get_field(self, scenario: Any, field_name: str, default: Any = None) -> Any:
        """Get a field from either a dict or a dataclass/object."""
        if isinstance(scenario, dict):
            return scenario.get(field_name, default)
        return getattr(scenario, field_name, default)

    def determine_role(self, scenario: Any) -> Role:
        """
        Determine the required role for a scenario.

        Accepts both dict and dataclass formats.
        """
        severity = str(self._get_field(scenario, "severity", "WARNING")).upper()
        estimated_cost = float(self._get_field(scenario, "estimated_cost", 0.0))
        kind = str(self._get_field(scenario, "kind", ""))

        # Rule 1: EMERGENCY severity always → EMERGENCY
        if severity == "EMERGENCY":
            return Role.EMERGENCY

        # Rule 2: Cost >= threshold → EMERGENCY
        if estimated_cost >= self.cost_threshold:
            return Role.EMERGENCY

        # Rule 3: Kind requires MANAGER
        if kind in KINDS_REQUIRING_MANAGER:
            return Role.MANAGER

        # Rule 4: CRITICAL severity with high cost
        if severity == "CRITICAL" and estimated_cost >= self.cost_threshold:
            return Role.EMERGENCY

        # Rule 5: Severity-based mapping
        role = _SEVERITY_ROLE.get(severity, Role.OPERATOR)

        # CRITICAL severity → at least MANAGER
        if severity == "CRITICAL":
            # Ensure at least MANAGER level
            if self.role_hierarchy.index(role) < self.role_hierarchy.index(Role.MANAGER):
                role = Role.MANAGER

        return role

    def determine_roles(self, scenarios: list[Any]) -> list[Role]:
        """Batch role determination for multiple scenarios."""
        return [self.determine_role(s) for s in scenarios]
