"""
Recommendation Engine (Phase 2C - Decision Engine Layer).

Generates DecisionEnvelope with 1-3 options based on diagnosis type and severity.

Rules:
  - UNKNOWN diagnosis -> 1 option (TECHNICAL_DISPATCH)
  - SENSOR_ANOMALY -> 2 options (OBSERVATION + TECHNICAL_DISPATCH)
  - PROCESS_ANOMALY high severity (RPN > 200) -> 1 option (STOP_PREP_ORDER)
  - PROCESS_ANOMALY medium severity (100 <= RPN <= 200) -> 3 options
  - PROCESS_ANOMALY low severity (RPN < 100) -> 2 options
  - Recommended marker = lowest E[cost]
  - Anti-pattern: 3+ same option cannot be recommended (CONTROLLED_STOP exempt)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_OPTIONS: int = 3
MIN_OPTIONS: int = 1
RPN_HIGH_THRESHOLD: int = 200
RPN_LOW_THRESHOLD: int = 100
ANTI_PATTERN_MAX_REPEATS: int = 3
CONTROLLED_STOP_EXEMPT: bool = True

# Default costs for work order types
DEFAULT_COSTS = {
    "OBSERVATION": 0.0,
    "TECHNICAL_DISPATCH": 15000.0,
    "SLOWDOWN_ORDER": 2500.0,
    "STOP_PREP_ORDER": 8000.0,
    "CONTROLLED_STOP": 25000.0,
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class WorkOrderType(str, Enum):
    """Work order types for decision options."""
    OBSERVATION = "OBSERVATION"
    TECHNICAL_DISPATCH = "TECHNICAL_DISPATCH"
    SLOWDOWN_ORDER = "SLOWDOWN_ORDER"
    STOP_PREP_ORDER = "STOP_PREP_ORDER"
    CONTROLLED_STOP = "CONTROLLED_STOP"


class DiagnosisType(str, Enum):
    """Diagnosis classification types."""
    UNKNOWN = "UNKNOWN"
    SENSOR_ANOMALY = "SENSOR_ANOMALY"
    PROCESS_ANOMALY = "PROCESS_ANOMALY"


class Severity(str, Enum):
    """Severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DecisionOption:
    """A single decision option."""
    work_order_type: WorkOrderType
    expected_cost: float
    is_recommended: bool = False
    description: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass
class DecisionEnvelope:
    """Envelope containing 1-3 decision options."""
    machine_id: str
    options: list[DecisionOption]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    diagnosis_type: str | None = None
    rpn: int = 0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """
    Generates decision options based on diagnosis type and severity.

    Follows rules:
    - UNKNOWN -> 1 option (TECHNICAL_DISPATCH)
    - SENSOR_ANOMALY -> 2 options (OBSERVATION + TECHNICAL_DISPATCH)
    - PROCESS_ANOMALY high (RPN > 200) -> 1 option (STOP_PREP_ORDER)
    - PROCESS_ANOMALY medium (100 <= RPN <= 200) -> 3 options
    - PROCESS_ANOMALY low (RPN < 100) -> 2 options
    """

    def __init__(self, cost_overrides: dict[str, float] | None = None):
        self.costs = dict(DEFAULT_COSTS)
        if cost_overrides:
            self.costs.update(cost_overrides)

    def generate(self, diagnosis: dict) -> DecisionEnvelope:
        """
        Generate a DecisionEnvelope with 1-3 options based on diagnosis.

        Args:
            diagnosis: Dict with keys:
                - diagnosis_type: DiagnosisType
                - machine_id: str
                - confidence: float
                - rpn: int
                - severity: Severity
                - mode_id: str (optional, for PROCESS_ANOMALY)

        Returns:
            DecisionEnvelope with options
        """
        diagnosis_type = diagnosis.get("diagnosis_type", DiagnosisType.UNKNOWN)
        machine_id = diagnosis.get("machine_id", "UNKNOWN")
        rpn = diagnosis.get("rpn", 0)
        severity = diagnosis.get("severity", Severity.LOW)
        confidence = diagnosis.get("confidence", 0.5)

        # Generate options based on diagnosis type
        if diagnosis_type == DiagnosisType.UNKNOWN:
            options = self._generate_unknown()
        elif diagnosis_type == DiagnosisType.SENSOR_ANOMALY:
            options = self._generate_sensor_anomaly()
        elif diagnosis_type == DiagnosisType.PROCESS_ANOMALY:
            options = self._generate_process_anomaly(rpn, severity)
        else:
            options = self._generate_unknown()

        # Validate options (anti-pattern guard)
        self._validate_options(options)

        # Mark recommended (lowest cost)
        self._mark_recommended(options)

        return DecisionEnvelope(
            machine_id=machine_id,
            options=options,
            timestamp=datetime.utcnow(),
            diagnosis_type=str(diagnosis_type),
            rpn=rpn,
            confidence=confidence,
        )

    def _generate_unknown(self) -> list[DecisionOption]:
        """UNKNOWN -> 1 option (TECHNICAL_DISPATCH)."""
        return [
            DecisionOption(
                work_order_type=WorkOrderType.TECHNICAL_DISPATCH,
                expected_cost=self.costs["TECHNICAL_DISPATCH"],
                description="Send technician for investigation",
            ),
        ]

    def _generate_sensor_anomaly(self) -> list[DecisionOption]:
        """SENSOR_ANOMALY -> 2 options (OBSERVATION + TECHNICAL_DISPATCH)."""
        return [
            DecisionOption(
                work_order_type=WorkOrderType.OBSERVATION,
                expected_cost=self.costs["OBSERVATION"],
                description="Continue monitoring sensor",
            ),
            DecisionOption(
                work_order_type=WorkOrderType.TECHNICAL_DISPATCH,
                expected_cost=self.costs["TECHNICAL_DISPATCH"],
                description="Inspect and calibrate sensor",
            ),
        ]

    def _generate_process_anomaly(
        self, rpn: int, severity: Severity
    ) -> list[DecisionOption]:
        """
        PROCESS_ANOMALY:
        - High (RPN > 200) -> 1 option (STOP_PREP_ORDER)
        - Medium (100 <= RPN <= 200) -> 3 options
        - Low (RPN < 100) -> 2 options
        """
        if rpn > RPN_HIGH_THRESHOLD:
            # High severity -> STOP_PREP_ORDER only
            return [
                DecisionOption(
                    work_order_type=WorkOrderType.STOP_PREP_ORDER,
                    expected_cost=self.costs["STOP_PREP_ORDER"],
                    description="Prepare for controlled stop",
                ),
            ]
        elif rpn >= RPN_LOW_THRESHOLD:
            # Medium severity -> 3 options
            return [
                DecisionOption(
                    work_order_type=WorkOrderType.OBSERVATION,
                    expected_cost=self.costs["OBSERVATION"],
                    description="Continue monitoring",
                ),
                DecisionOption(
                    work_order_type=WorkOrderType.SLOWDOWN_ORDER,
                    expected_cost=self.costs["SLOWDOWN_ORDER"],
                    description="Reduce load to slow degradation",
                ),
                DecisionOption(
                    work_order_type=WorkOrderType.STOP_PREP_ORDER,
                    expected_cost=self.costs["STOP_PREP_ORDER"],
                    description="Prepare for controlled stop",
                ),
            ]
        else:
            # Low severity -> 2 options
            return [
                DecisionOption(
                    work_order_type=WorkOrderType.OBSERVATION,
                    expected_cost=self.costs["OBSERVATION"],
                    description="Continue monitoring",
                ),
                DecisionOption(
                    work_order_type=WorkOrderType.TECHNICAL_DISPATCH,
                    expected_cost=self.costs["TECHNICAL_DISPATCH"],
                    description="Schedule technician inspection",
                ),
            ]

    def _validate_options(self, options: list[DecisionOption]) -> None:
        """
        Anti-pattern guard: 3+ same option cannot be recommended.
        CONTROLLED_STOP is exempt.

        Raises ValueError if anti-pattern violated.
        """
        type_counts: dict[WorkOrderType, int] = {}
        for opt in options:
            wtype = opt.work_order_type
            type_counts[wtype] = type_counts.get(wtype, 0) + 1

        for wtype, count in type_counts.items():
            if wtype == WorkOrderType.CONTROLLED_STOP and CONTROLLED_STOP_EXEMPT:
                continue
            if count >= ANTI_PATTERN_MAX_REPEATS:
                raise ValueError(
                    f"Anti-pattern violation: {wtype.value} appears {count} times "
                    f"(max {ANTI_PATTERN_MAX_REPEATS - 1})"
                )

    def _mark_recommended(self, options: list[DecisionOption]) -> None:
        """Mark the option with lowest expected_cost as recommended."""
        if not options:
            return

        # Find minimum cost
        min_cost = min(o.expected_cost for o in options)

        # Mark only the first option with min cost as recommended
        marked = False
        for opt in options:
            if opt.expected_cost == min_cost and not marked:
                opt.is_recommended = True
                marked = True
            else:
                opt.is_recommended = False
