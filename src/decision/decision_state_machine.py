"""
Decision State Machine (Phase 2B - Core Logic Layer).

7-state machine for maintenance decision lifecycle:
  HEALTHY → MONITORING → ALARMED → MONITOR_30M → {ESCALATED | RECOVERED}
  RECOVERED → STABLE → HEALTHY

Features:
- 30-minute watchdog timer (MONITOR_30M → ESCALATED on expiry)
- Anti-pattern guard (EEMUA 191: max 3 same option, CONTROLLED_STOP exempt)
- Time-based checks for STABLE and HEALTHY transitions
- Full state history tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WATCHDOG_DURATION_MINUTES: int = 30
MAX_OPTION_REPEATS: int = 3
STABLE_DURATION_MINUTES: int = 15
HEALTHY_DURATION_MINUTES: int = 30
CONTROLLED_STOP_OPTION: str = "CONTROLLED_STOP"


# ---------------------------------------------------------------------------
# Enums & Exceptions
# ---------------------------------------------------------------------------
class DecisionState(str, Enum):
    """Seven states in the decision lifecycle."""
    HEALTHY = "HEALTHY"
    MONITORING = "MONITORING"
    ALARMED = "ALARMED"
    MONITOR_30M = "MONITOR_30M"
    ESCALATED = "ESCALATED"
    RECOVERED = "RECOVERED"
    STABLE = "STABLE"


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class AntiPatternViolation(Exception):  # noqa: N818 — public API name, kept stable
    """Raised when anti-pattern guard is violated (EEMUA 191)."""
    pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class OptionRecord:
    """Record of an option recommendation."""
    option_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class _StateEntry:
    """Internal state history entry."""
    state: DecisionState
    entered_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# WatchdogTimer
# ---------------------------------------------------------------------------
class WatchdogTimer:
    """30-minute watchdog timer for MONITOR_30M state."""

    def __init__(self, duration_minutes: int = WATCHDOG_DURATION_MINUTES):
        self._duration = timedelta(minutes=duration_minutes)
        self._deadline: datetime | None = None
        self._active: bool = False

    def start(self, now: datetime | None = None) -> None:
        """Start the watchdog timer."""
        self._deadline = (now or datetime.utcnow()) + self._duration
        self._active = True

    def stop(self) -> None:
        """Stop the watchdog timer."""
        self._deadline = None
        self._active = False

    @property
    def active(self) -> bool:
        """Whether the watchdog is currently active."""
        return self._active

    @property
    def deadline(self) -> datetime | None:
        """The deadline datetime, or None if not active."""
        return self._deadline

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if the watchdog has expired."""
        if not self._active or self._deadline is None:
            return False
        check_time = now or datetime.utcnow()
        return check_time >= self._deadline


# ---------------------------------------------------------------------------
# DecisionStateMachine
# ---------------------------------------------------------------------------

# Allowed transitions: from_state → set of valid to_states
_TRANSITIONS: dict[DecisionState, set] = {
    DecisionState.HEALTHY: {DecisionState.MONITORING},
    DecisionState.MONITORING: {DecisionState.ALARMED},
    DecisionState.ALARMED: {DecisionState.MONITOR_30M},
    DecisionState.MONITOR_30M: {DecisionState.ESCALATED, DecisionState.RECOVERED},
    DecisionState.ESCALATED: {DecisionState.RECOVERED},
    DecisionState.RECOVERED: {DecisionState.STABLE},
    DecisionState.STABLE: {DecisionState.HEALTHY},
}


class DecisionStateMachine:
    """
    7-state decision machine with watchdog and anti-pattern guard.

    Lifecycle:
        HEALTHY → MONITORING → ALARMED → MONITOR_30M → RECOVERED → STABLE → HEALTHY
                                                ↓
                                            ESCALATED → RECOVERED
    """

    def __init__(self, machine_id: str):
        if not machine_id:
            raise ValueError("machine_id cannot be empty or None")

        self._machine_id = machine_id
        self._state = DecisionState.HEALTHY
        self._state_history: list[_StateEntry] = [
            _StateEntry(state=DecisionState.HEALTHY)
        ]
        self._watchdog = WatchdogTimer()
        self._option_counts: dict[str, int] = {}
        self._option_records: list[OptionRecord] = []

    @property
    def machine_id(self) -> str:
        return self._machine_id

    @property
    def state(self) -> DecisionState:
        return self._state

    @property
    def state_history(self) -> list[_StateEntry]:
        return list(self._state_history)

    @property
    def watchdog_active(self) -> bool:
        return self._watchdog.active

    @property
    def watchdog_deadline(self) -> datetime | None:
        return self._watchdog.deadline

    def transition(self, new_state: DecisionState) -> None:
        """
        Transition to a new state.

        Raises:
            InvalidTransitionError: if the transition is not allowed.
        """
        allowed = _TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.value} to {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        self._state = new_state
        self._state_history.append(_StateEntry(state=new_state))

        # Handle watchdog
        if new_state == DecisionState.MONITOR_30M:
            self._watchdog.start()
        elif new_state in (DecisionState.RECOVERED, DecisionState.ESCALATED):
            self._watchdog.stop()

        # Reset option counters on state change
        self._option_counts.clear()

    def recommend_option(self, option_name: str) -> OptionRecord:
        """
        Recommend an option (with anti-pattern guard).

        Same option can be recommended at most MAX_OPTION_REPEATS times.
        CONTROLLED_STOP is exempt from this limit.

        Raises:
            AntiPatternViolation: if the option has been recommended too many times.
        """
        # CONTROLLED_STOP is always allowed
        if option_name == CONTROLLED_STOP_OPTION:
            record = OptionRecord(option_name=option_name)
            self._option_records.append(record)
            return record

        current_count = self._option_counts.get(option_name, 0)
        if current_count >= MAX_OPTION_REPEATS:
            raise AntiPatternViolation(
                f"Option '{option_name}' has already been recommended "
                f"{current_count} times (max: {MAX_OPTION_REPEATS}). "
                f"EEMUA 191 anti-pattern guard."
            )

        self._option_counts[option_name] = current_count + 1
        record = OptionRecord(option_name=option_name)
        self._option_records.append(record)
        return record

    def watchdog_check(self, now: datetime | None = None) -> DecisionState | None:
        """
        Check if the watchdog has expired.

        Returns:
            DecisionState.ESCALATED if expired, None otherwise.
        """
        if self._state != DecisionState.MONITOR_30M:
            return None

        if self._watchdog.is_expired(now):
            return DecisionState.ESCALATED

        return None

    def stable_due(self, now: datetime | None = None) -> bool:
        """
        Check if enough time has passed in RECOVERED to transition to STABLE.
        """
        if self._state != DecisionState.RECOVERED:
            return False

        check_time = now or datetime.utcnow()
        # Find when we entered RECOVERED
        entered_at = self._get_state_entered_at(DecisionState.RECOVERED)
        if entered_at is None:
            return False

        elapsed = check_time - entered_at
        return elapsed >= timedelta(minutes=STABLE_DURATION_MINUTES)

    def healthy_due(self, now: datetime | None = None) -> bool:
        """
        Check if enough time has passed in STABLE to transition to HEALTHY.
        """
        if self._state != DecisionState.STABLE:
            return False

        check_time = now or datetime.utcnow()
        entered_at = self._get_state_entered_at(DecisionState.STABLE)
        if entered_at is None:
            return False

        elapsed = check_time - entered_at
        return elapsed >= timedelta(minutes=HEALTHY_DURATION_MINUTES)

    def reset_alarm(self) -> None:
        """
        Reset alarm state: return to HEALTHY, stop watchdog, clear option history.
        """
        self._state = DecisionState.HEALTHY
        self._watchdog.stop()
        self._option_counts.clear()
        self._option_records.clear()
        self._state_history.append(_StateEntry(state=DecisionState.HEALTHY))

    def _get_state_entered_at(self, state: DecisionState) -> datetime | None:
        """Find when we last entered a given state."""
        for entry in reversed(self._state_history):
            if entry.state == state:
                return entry.entered_at
        return None
