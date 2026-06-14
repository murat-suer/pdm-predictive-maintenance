"""
PDM v3 Decision Module.

Provides:
- EscalationPolicy: Role-based escalation logic
- AlarmStateMachine: ISA-18.2 compliant state management
- ShiftCalendar: Shift-aware maintenance window recommendation
- FailureModeLibrary: YAML-based FMEA failure mode matching
"""

from src.decision.alarm_state_machine import (
    MAX_SHELVE_DURATION_HOURS,
    MAX_SUPPRESS_DURATION_DAYS,
    AlarmState,
    AlarmStateMachine,
    AuditEntry,
    TransitionError,
)
from src.decision.alarm_state_machine import (
    Role as AlarmRole,
)
from src.decision.escalation import (
    EMERGENCY_THRESHOLD,
    KINDS_REQUIRING_MANAGER,
    EscalationPolicy,
    Scenario,
)
from src.decision.escalation import (
    Role as EscalationRole,
)
from src.decision.failure_mode_library import (
    FailureMode,
    FailureModeLibrary,
    MatchResult,
    SignatureRule,
)
from src.decision.shift_awareness import (
    MaintenanceWindow,
    Shift,
    ShiftCalendar,
    WindowRecommendation,
)

__all__ = [
    # Escalation
    "EscalationPolicy",
    "EscalationRole",
    "Scenario",
    "EMERGENCY_THRESHOLD",
    "KINDS_REQUIRING_MANAGER",
    # Alarm State Machine
    "AlarmStateMachine",
    "AlarmState",
    "AlarmRole",
    "TransitionError",
    "AuditEntry",
    "MAX_SHELVE_DURATION_HOURS",
    "MAX_SUPPRESS_DURATION_DAYS",
    # Shift Awareness
    "ShiftCalendar",
    "Shift",
    "MaintenanceWindow",
    "WindowRecommendation",
    # Failure Mode Library
    "FailureModeLibrary",
    "FailureMode",
    "SignatureRule",
    "MatchResult",
]
