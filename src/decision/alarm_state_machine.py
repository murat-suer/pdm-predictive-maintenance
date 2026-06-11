"""
ISA-18.2 compliant alarm state machine.

Manages alarm lifecycle: NORMAL → UNACKNOWLEDGED → ACKNOWLEDGED → NORMAL,
with SHELVED, OUT_OF_SERVICE, and SUPPRESSED side states.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class AlarmState(str, Enum):
    """ISA-18.2 alarm states."""
    NORMAL = "NORMAL"
    UNACKNOWLEDGED = "UNACKNOWLEDGED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    SHELVED = "SHELVED"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"
    SUPPRESSED = "SUPPRESSED"


class Role(str, Enum):
    """Operator roles with hierarchy."""
    OPERATOR = "OPERATOR"
    SUPERVISOR = "SUPERVISOR"
    MANAGER = "MANAGER"

    def __lt__(self, other):
        order = [Role.OPERATOR, Role.SUPERVISOR, Role.MANAGER]
        return order.index(self) < order.index(other)

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class TransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


@dataclass
class AuditEntry:
    """Audit trail entry for state transitions."""
    from_state: AlarmState
    to_state: AlarmState
    role: Role
    timestamp: datetime
    reason: str | None = None
    work_order_id: str | None = None


# Maximum shelve duration in hours
MAX_SHELVE_DURATION_HOURS = 24.0

# Maximum suppress duration before mandatory review (days)
MAX_SUPPRESS_DURATION_DAYS = 30

# Valid state transitions (from → set of valid targets)
VALID_TRANSITIONS = {
    AlarmState.NORMAL: {AlarmState.UNACKNOWLEDGED, AlarmState.SHELVED, AlarmState.OUT_OF_SERVICE, AlarmState.SUPPRESSED},
    AlarmState.UNACKNOWLEDGED: {AlarmState.ACKNOWLEDGED, AlarmState.NORMAL},
    AlarmState.ACKNOWLEDGED: {AlarmState.NORMAL, AlarmState.SHELVED, AlarmState.OUT_OF_SERVICE},
    AlarmState.SHELVED: {AlarmState.NORMAL},
    AlarmState.OUT_OF_SERVICE: {AlarmState.NORMAL},
    AlarmState.SUPPRESSED: {AlarmState.NORMAL},
}


class AlarmStateMachine:
    """
    ISA-18.2 compliant alarm state machine.

    Manages state transitions with role-based permissions and audit trail.
    """

    def __init__(self, alarm_id: str, machine_id: str):
        self.alarm_id = alarm_id
        self.machine_id = machine_id
        self._state: AlarmState = AlarmState.NORMAL
        self._audit_trail: list[AuditEntry] = []
        self._shelve_expiry: datetime | None = None
        self._work_order_id: str | None = None
        self._review_date: datetime | None = None

    @property
    def state(self) -> AlarmState:
        """Current alarm state."""
        return self._state

    @property
    def audit_trail(self) -> list[AuditEntry]:
        """Complete audit trail of all transitions."""
        return self._audit_trail

    @property
    def work_order_id(self) -> str | None:
        """Current work order ID (if OUT_OF_SERVICE)."""
        return self._work_order_id

    @property
    def review_date(self) -> datetime | None:
        """Review date for SUPPRESSED state."""
        return self._review_date

    def _add_audit(self, from_state: AlarmState, to_state: AlarmState,
                   role: Role, reason: str | None = None,
                   work_order_id: str | None = None):
        """Add an audit entry."""
        entry = AuditEntry(
            from_state=from_state,
            to_state=to_state,
            role=role,
            timestamp=datetime.utcnow(),
            reason=reason,
            work_order_id=work_order_id,
        )
        self._audit_trail.append(entry)

    def transition_to(self, target_state: AlarmState, role: Role):
        """
        Transition to a target state.

        Validates the transition against VALID_TRANSITIONS.
        """
        if target_state not in VALID_TRANSITIONS.get(self._state, set()):
            raise TransitionError(
                f"Invalid transition: {self._state.value} → {target_state.value}"
            )

        from_state = self._state
        self._state = target_state
        self._add_audit(from_state, target_state, role)

    def shelve(self, role: Role, reason: str, duration_hours: float):
        """
        Shelve the alarm (suppress notifications temporarily).

        Requirements:
        - role >= SUPERVISOR
        - reason is mandatory (non-empty)
        - duration_hours <= MAX_SHELVE_DURATION_HOURS
        - Current state must allow transition to SHELVED
        """
        # Permission check
        if role < Role.SUPERVISOR:
            raise PermissionError(
                f"Role {role.value} cannot shelve alarms. Requires SUPERVISOR or higher."
            )

        # Reason validation
        if not reason or not reason.strip():
            raise ValueError("Shelve reason is mandatory.")

        # Duration validation
        if duration_hours > MAX_SHELVE_DURATION_HOURS:
            raise ValueError(
                f"Shelve duration {duration_hours}h exceeds maximum {MAX_SHELVE_DURATION_HOURS}h."
            )

        # State validation
        if AlarmState.SHELVED not in VALID_TRANSITIONS.get(self._state, set()):
            raise TransitionError(
                f"Cannot shelve from state {self._state.value}."
            )

        from_state = self._state
        self._state = AlarmState.SHELVED
        self._shelve_expiry = datetime.utcnow() + timedelta(hours=duration_hours)
        self._add_audit(from_state, AlarmState.SHELVED, role, reason=reason)

    def unshelve(self, role: Role):
        """Return from SHELVED to NORMAL."""
        if self._state != AlarmState.SHELVED:
            raise TransitionError(
                f"Cannot unshelve from state {self._state.value}."
            )

        from_state = self._state
        self._state = AlarmState.NORMAL
        self._shelve_expiry = None
        self._add_audit(from_state, AlarmState.NORMAL, role)

    def set_out_of_service(self, role: Role, work_order_id: str):
        """
        Set alarm OUT_OF_SERVICE.

        Requirements:
        - role >= MANAGER
        - work_order_id is mandatory (non-empty)
        """
        # Work order validation
        if not work_order_id or not work_order_id.strip():
            raise ValueError("work_order_id is mandatory for OUT_OF_SERVICE.")

        # State validation
        if AlarmState.OUT_OF_SERVICE not in VALID_TRANSITIONS.get(self._state, set()):
            raise TransitionError(
                f"Cannot set OUT_OF_SERVICE from state {self._state.value}."
            )

        from_state = self._state
        self._state = AlarmState.OUT_OF_SERVICE
        self._work_order_id = work_order_id
        self._add_audit(from_state, AlarmState.OUT_OF_SERVICE, role,
                        work_order_id=work_order_id)

    def return_to_service(self, role: Role):
        """Return from OUT_OF_SERVICE to NORMAL."""
        if self._state != AlarmState.OUT_OF_SERVICE:
            raise TransitionError(
                f"Cannot return to service from state {self._state.value}."
            )

        from_state = self._state
        self._state = AlarmState.NORMAL
        self._work_order_id = None
        self._add_audit(from_state, AlarmState.NORMAL, role)

    def suppress(self, role: Role, reason: str):
        """
        Suppress the alarm (Manager only, 30-day review).

        Requirements:
        - role == MANAGER
        - reason is mandatory (non-empty)
        """
        # Permission check - MANAGER only
        if role < Role.MANAGER:
            raise PermissionError(
                f"Role {role.value} cannot suppress alarms. Requires MANAGER."
            )

        # Reason validation
        if not reason or not reason.strip():
            raise ValueError("Suppress reason is mandatory.")

        # State validation
        if AlarmState.SUPPRESSED not in VALID_TRANSITIONS.get(self._state, set()):
            raise TransitionError(
                f"Cannot suppress from state {self._state.value}."
            )

        from_state = self._state
        self._state = AlarmState.SUPPRESSED
        self._review_date = datetime.utcnow() + timedelta(days=MAX_SUPPRESS_DURATION_DAYS)
        self._add_audit(from_state, AlarmState.SUPPRESSED, role, reason=reason)

    def check_expiry(self, now: datetime | None = None):
        """
        Check if shelve has expired and auto-return to NORMAL.

        Args:
            now: Current time (defaults to datetime.utcnow())
        """
        if now is None:
            now = datetime.utcnow()

        if self._state == AlarmState.SHELVED and self._shelve_expiry is not None:
            if now >= self._shelve_expiry:
                self._state = AlarmState.NORMAL
                self._shelve_expiry = None
