"""
Escalation Manager (Phase 2B - Core Logic Layer).

L1→L2→L3 escalation with:
- Trend analysis (IMPROVING / STABLE / WORSENING)
- 15-minute L2→L3 timeout
- Auto-resolve on 3+ consecutive IMPROVING checks
- Score history (FIFO, max 5 entries)
- Downtime simulation (Gaussian per machine type)
- DEMO_MODE auto-acknowledge
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
L2_TIMEOUT_MINUTES: int = 15
AUTO_RESOLVE_IMPROVING_COUNT: int = 3
SCORE_HISTORY_SIZE: int = 5
DEMO_MODE: bool = False

# Downtime parameters: (mean_minutes, std_minutes) per machine type
_DOWNTIME_PARAMS: dict[str, tuple[float, float]] = {
    "compressor": (90.0, 15.0),
    "pump": (60.0, 10.0),
    "fan": (45.0, 8.0),
    "motor": (75.0, 12.0),
    "conveyor": (50.0, 10.0),
}
_DEFAULT_DOWNTIME_PARAMS: tuple[float, float] = (60.0, 15.0)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class EscalationLevel(str, Enum):
    """Escalation levels."""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class Trend(str, Enum):
    """Trend classification."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    WORSENING = "WORSENING"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScoreEntry:
    """A single anomaly score entry with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EscalationRecord:
    """Record of an escalation event."""
    level: EscalationLevel
    reason: str
    escalated_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# EscalationManager
# ---------------------------------------------------------------------------
class EscalationManager:
    """
    L1→L2→L3 escalation manager with trend analysis and auto-resolve.

    Features:
    - Score history (FIFO, capped at SCORE_HISTORY_SIZE)
    - Trend classification (IMPROVING/STABLE/WORSENING)
    - L2→L3 timeout after L2_TIMEOUT_MINUTES
    - Auto-resolve after AUTO_RESOLVE_IMPROVING_COUNT consecutive IMPROVING
    - Downtime simulation (Gaussian per machine type)
    - DEMO_MODE auto-acknowledge
    """

    def __init__(
        self,
        machine_id: str,
        demo_mode: bool = False,
    ):
        if not machine_id:
            raise ValueError("machine_id cannot be empty or None")

        self._machine_id = machine_id
        self._demo_mode = demo_mode
        self._current_level: EscalationLevel | None = None
        self._score_history: list[ScoreEntry] = []
        self._escalation_history: list[EscalationRecord] = []
        self._improving_streak: int = 0
        self._last_trend: Trend | None = None
        self._resolved: bool = False

    @property
    def machine_id(self) -> str:
        return self._machine_id

    @property
    def current_level(self) -> EscalationLevel | None:
        return self._current_level

    @property
    def score_history(self) -> list[ScoreEntry]:
        return list(self._score_history)

    @property
    def escalation_history(self) -> list[EscalationRecord]:
        return list(self._escalation_history)

    @property
    def is_resolved(self) -> bool:
        return self._resolved

    def escalate(self, level: EscalationLevel, reason: str) -> None:
        """
        Escalate to a new level.

        Raises:
            ValueError: if reason is empty
        """
        if not reason:
            raise ValueError("Escalation reason cannot be empty")

        record = EscalationRecord(
            level=level,
            reason=reason,
            escalated_at=datetime.utcnow(),
            acknowledged=self._demo_mode,  # Auto-ack in DEMO_MODE
        )

        self._current_level = level
        self._escalation_history.append(record)
        self._resolved = False

    def get_current_record(self) -> EscalationRecord | None:
        """Get the most recent escalation record."""
        if not self._escalation_history:
            return None
        return self._escalation_history[-1]

    def record_score(self, value: float) -> None:
        """
        Record an anomaly score.

        History is capped at SCORE_HISTORY_SIZE (FIFO).
        Values must be in [0.0, 1.0] range; raises ValueError otherwise.
        """
        # Validate score range
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Score must be in [0.0, 1.0] range, got {value}")

        entry = ScoreEntry(value=value, timestamp=datetime.utcnow())
        self._score_history.append(entry)

        # FIFO eviction
        if len(self._score_history) > SCORE_HISTORY_SIZE:
            self._score_history = self._score_history[-SCORE_HISTORY_SIZE:]

        # Update improving streak
        self._update_improving_streak()

    def _update_improving_streak(self) -> None:
        """Update the consecutive IMPROVING streak counter.

        Counts the number of scores in the current improving run.
        A run starts with the first score (streak=1) and continues
        as long as each new score is strictly less than the previous.
        """
        if len(self._score_history) < 2:
            # First score: streak = 1
            self._improving_streak = 1 if self._score_history else 0
            return

        latest = self._score_history[-1].value
        previous = self._score_history[-2].value

        if latest < previous:
            # Continuing an improving run
            self._improving_streak += 1
        else:
            # Run broken (equal or worsening) - restart with this score
            self._improving_streak = 1

    def classify_trend(self) -> Trend:
        """
        Classify the trend based on score history.

        Uses first-half vs second-half comparison.

        Returns:
            Trend.IMPROVING if scores are decreasing
            Trend.WORSENING if scores are increasing
            Trend.STABLE if scores are constant or insufficient data
        """
        if len(self._score_history) < 2:
            return Trend.STABLE

        values = [e.value for e in self._score_history]
        return self._classify_from_values(values)

    def _classify_from_values(self, values: list[float]) -> Trend:
        """Classify trend from a list of values."""
        if len(values) < 2:
            return Trend.STABLE

        # Count increases vs decreases
        increases = 0
        decreases = 0
        for i in range(1, len(values)):
            if values[i] > values[i - 1]:
                increases += 1
            elif values[i] < values[i - 1]:
                decreases += 1

        total_changes = increases + decreases
        if total_changes == 0:
            return Trend.STABLE

        # Majority vote
        if decreases > increases:
            return Trend.IMPROVING
        elif increases > decreases:
            return Trend.WORSENING
        else:
            return Trend.STABLE

    def check_timeout(self, now: datetime | None = None) -> bool:
        """
        Check if L2 timeout has expired (should escalate to L3).

        Returns:
            True if timeout expired (should escalate to L3), False otherwise.
        """
        if self._current_level != EscalationLevel.L2:
            return False

        # Find when L2 was escalated
        l2_record = None
        for record in reversed(self._escalation_history):
            if record.level == EscalationLevel.L2:
                l2_record = record
                break

        if l2_record is None:
            return False

        check_time = now or datetime.utcnow()
        elapsed = check_time - l2_record.escalated_at
        return elapsed >= timedelta(minutes=L2_TIMEOUT_MINUTES)

    def check_auto_resolve(self) -> bool:
        """
        Check if auto-resolve should trigger (3+ consecutive IMPROVING).

        Returns:
            True if auto-resolve should trigger.
        """
        trend = self.classify_trend()
        if trend != Trend.IMPROVING:
            return False

        return self._improving_streak >= AUTO_RESOLVE_IMPROVING_COUNT

    def resolve(self) -> None:
        """Resolve the current escalation."""
        self._resolved = True
        self._current_level = None

    def simulate_downtime(
        self,
        machine_type: str,
        seed: int | None = None,
    ) -> float:
        """
        Simulate downtime using Gaussian distribution per machine type.

        Args:
            machine_type: Type of machine (compressor, pump, fan, etc.)
            seed: Optional random seed for reproducibility

        Returns:
            Simulated downtime in minutes (always positive)
        """
        params = _DOWNTIME_PARAMS.get(machine_type, _DEFAULT_DOWNTIME_PARAMS)
        mean, std = params

        rng = random.Random(seed)
        value = rng.gauss(mean, std)

        # Ensure positive
        return max(1.0, value)
