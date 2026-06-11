"""
Hybrid rule+ML diagnostic engine (Phase 2B - Core Logic Layer).

Combines FailureModeLibrary (rule-based matching) with an ML anomaly scorer
to produce three diagnosis types:
  - PROCESS_ANOMALY: library match + high ML score
  - SENSOR_ANOMALY: recurring pattern on same sensor (3+ times)
  - UNKNOWN: ML detects anomaly but no library match

Reliability scoring: 0.6 * signature_confidence + 0.4 * ml_score
Sensor attribution: SHAP-like weighted contributions (normalized to sum=1.0)
Evidence chain: ordered audit trail of diagnostic steps.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.decision.failure_mode_library import FailureModeLibrary, MatchResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RELIABILITY_RULE_WEIGHT: float = 0.6
RELIABILITY_ML_WEIGHT: float = 0.4
SENSOR_ANOMALY_RECURRING_THRESHOLD: int = 3
ML_ANOMALY_THRESHOLD: float = 0.5
LIBRARY_CONFIDENCE_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------
class DiagnosisType(str, Enum):
    """Three possible diagnosis outcomes."""
    PROCESS_ANOMALY = "PROCESS_ANOMALY"
    SENSOR_ANOMALY = "SENSOR_ANOMALY"
    UNKNOWN = "UNKNOWN"


@dataclass
class SensorContribution:
    """SHAP-like sensor attribution."""
    sensor_name: str
    weight: float


@dataclass
class EvidenceItem:
    """Single evidence item in the diagnostic audit trail."""
    timestamp: datetime
    source: str  # "rule_engine", "ml_model", "sensor_data", "history"
    description: str
    details: dict[str, Any] | None = None


@dataclass
class DiagnosisResult:
    """Complete diagnosis result."""
    diagnosis_type: DiagnosisType
    machine_id: str
    mode_id: str | None = None
    mode_name: str | None = None
    confidence: float = 0.0
    signature_confidence: float = 0.0
    ml_score: float = 0.0
    reliability: float = 0.0
    sensor_contributions: list[SensorContribution] = field(default_factory=list)
    evidence_chain: list[EvidenceItem] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@runtime_checkable
class MLScorer(Protocol):
    """Protocol for ML anomaly scoring."""

    def score(self, sensor_data: dict[str, Any]) -> float:
        """Return anomaly score between 0.0 and 1.0."""
        ...


# ---------------------------------------------------------------------------
# DiagnosticEngine
# ---------------------------------------------------------------------------
class DiagnosticEngine:
    """
    Hybrid rule+ML diagnostic engine.

    Workflow:
    1. ML score < threshold → UNKNOWN (early exit)
    2. Library match (failure_mode_library)
    3. Recurring pattern detection → SENSOR_ANOMALY
    4. PROCESS_ANOMALY vs UNKNOWN (library threshold)
    """

    def __init__(
        self,
        ml_scorer: MLScorer | None = None,
        library: FailureModeLibrary | None = None,
        ml_threshold: float = ML_ANOMALY_THRESHOLD,
        library_threshold: float = LIBRARY_CONFIDENCE_THRESHOLD,
    ):
        self._ml_scorer = ml_scorer
        # Auto-create default library if none provided
        if library is not None:
            self._library = library
        else:
            try:
                self._library = FailureModeLibrary()
            except (FileNotFoundError, ValueError):
                self._library = None
        self._ml_threshold = ml_threshold
        self._library_threshold = library_threshold
        # Track recurring sensor events: key = (sensor_name, machine_id)
        self._sensor_events: dict[tuple, int] = defaultdict(int)

    def record_sensor_event(
        self,
        sensor_name: str,
        machine_id: str,
        event_type: str,
    ) -> None:
        """Record a sensor event for recurring pattern detection."""
        key = (sensor_name, machine_id)
        self._sensor_events[key] += 1

    def _get_recurring_count(self, sensor_name: str, machine_id: str) -> int:
        """Get the count of events for a sensor on a machine."""
        key = (sensor_name, machine_id)
        return self._sensor_events.get(key, 0)

    def diagnose(
        self,
        sensor_data: dict[str, Any],
        machine_id: str,
    ) -> DiagnosisResult:
        """
        Run full diagnostic pipeline.

        Args:
            sensor_data: Sensor readings dict
            machine_id: Machine identifier

        Returns:
            DiagnosisResult with type, confidence, reliability, etc.

        Raises:
            TypeError: if sensor_data is None
            ValueError: if machine_id is empty
        """
        # Validation
        if sensor_data is None:
            raise TypeError("sensor_data cannot be None")
        if not machine_id:
            raise ValueError("machine_id cannot be empty")

        evidence_chain: list[EvidenceItem] = []
        now = datetime.utcnow()

        # Record sensor data evidence
        evidence_chain.append(EvidenceItem(
            timestamp=now,
            source="sensor_data",
            description=f"Received {len(sensor_data)} sensor readings for {machine_id}",
            details={"sensor_count": len(sensor_data)},
        ))

        # Step 1: Get ML anomaly score
        ml_score = 0.0
        if self._ml_scorer is not None:
            ml_score = self._ml_scorer.score(sensor_data)
            evidence_chain.append(EvidenceItem(
                timestamp=now,
                source="ml_model",
                description=f"ML anomaly score: {ml_score:.4f}",
                details={"ml_score": ml_score},
            ))

        # Step 2: Check for recurring sensor pattern (SENSOR_ANOMALY)
        # Check if any sensor in the data has reached recurring threshold
        recurring_sensor = None
        for sensor_name in sensor_data:
            # Include current reading as an event
            info = sensor_data[sensor_name]
            if isinstance(info, dict) and info.get("trend") == "up":
                count = self._get_recurring_count(sensor_name, machine_id)
                # +1 for current diagnosis
                if count + 1 >= SENSOR_ANOMALY_RECURRING_THRESHOLD:
                    recurring_sensor = sensor_name
                    break

        if recurring_sensor is not None:
            evidence_chain.append(EvidenceItem(
                timestamp=now,
                source="history",
                description=f"Recurring pattern detected on sensor '{recurring_sensor}'",
                details={"sensor": recurring_sensor, "count": self._get_recurring_count(recurring_sensor, machine_id) + 1},
            ))

            # Build sensor contributions
            sensor_contributions = self._compute_sensor_contributions(sensor_data)

            return DiagnosisResult(
                diagnosis_type=DiagnosisType.SENSOR_ANOMALY,
                machine_id=machine_id,
                mode_id=None,
                mode_name=None,
                confidence=0.0,
                signature_confidence=0.0,
                ml_score=ml_score,
                reliability=self.calculate_reliability(0.0, ml_score),
                sensor_contributions=sensor_contributions,
                evidence_chain=evidence_chain,
                timestamp=now,
            )

        # Step 3: Library matching
        matches: list[MatchResult] = []
        if self._library is not None and sensor_data:
            matches = self._library.match(sensor_data, machine_id)
            evidence_chain.append(EvidenceItem(
                timestamp=now,
                source="rule_engine",
                description=f"Library matching: {len(matches)} matches found",
                details={"match_count": len(matches)},
            ))

        # Step 4: Determine diagnosis type
        # Select best match by confidence (not confidence*RPN)
        best_match = max(matches, key=lambda m: m.confidence) if matches else None

        if best_match is not None and best_match.confidence >= self._library_threshold and ml_score >= self._ml_threshold:
            # PROCESS_ANOMALY: library match + high ML score
            signature_confidence = best_match.confidence
            reliability = self.calculate_reliability(signature_confidence, ml_score)
            sensor_contributions = self._compute_sensor_contributions(sensor_data)

            evidence_chain.append(EvidenceItem(
                timestamp=now,
                source="rule_engine",
                description=f"PROCESS_ANOMALY: matched mode '{best_match.mode_id}' with confidence {signature_confidence:.4f}",
                details={"mode_id": best_match.mode_id, "confidence": signature_confidence},
            ))

            return DiagnosisResult(
                diagnosis_type=DiagnosisType.PROCESS_ANOMALY,
                machine_id=machine_id,
                mode_id=best_match.mode_id,
                mode_name=best_match.description,
                confidence=signature_confidence,
                signature_confidence=signature_confidence,
                ml_score=ml_score,
                reliability=reliability,
                sensor_contributions=sensor_contributions,
                evidence_chain=evidence_chain,
                timestamp=now,
            )
        else:
            # UNKNOWN: ML detects anomaly but no library match (or below threshold)
            # signature_confidence is 0.0 for UNKNOWN (no confirmed failure mode)
            signature_confidence = 0.0
            reliability = self.calculate_reliability(signature_confidence, ml_score)
            sensor_contributions = self._compute_sensor_contributions(sensor_data)

            evidence_chain.append(EvidenceItem(
                timestamp=now,
                source="rule_engine",
                description="UNKNOWN: no sufficient library match",
                details={"best_match": best_match.mode_id if best_match else None},
            ))

            return DiagnosisResult(
                diagnosis_type=DiagnosisType.UNKNOWN,
                machine_id=machine_id,
                mode_id=best_match.mode_id if best_match else None,
                mode_name=best_match.description if best_match else None,
                confidence=signature_confidence,
                signature_confidence=signature_confidence,
                ml_score=ml_score,
                reliability=reliability,
                sensor_contributions=sensor_contributions,
                evidence_chain=evidence_chain,
                timestamp=now,
            )

    def calculate_reliability(
        self,
        signature_confidence: float,
        ml_score: float,
    ) -> float:
        """
        Calculate reliability score.

        Formula: 0.6 * signature_confidence + 0.4 * ml_score
        """
        return (
            RELIABILITY_RULE_WEIGHT * signature_confidence
            + RELIABILITY_ML_WEIGHT * ml_score
        )

    def _compute_sensor_contributions(
        self,
        sensor_data: dict[str, Any],
    ) -> list[SensorContribution]:
        """
        Compute SHAP-like sensor contributions (normalized weights).

        Uses delta_pct if available, otherwise computes from value/baseline.
        Falls back to equal distribution if no deviation info.
        """
        if not sensor_data:
            return []

        raw_weights: dict[str, float] = {}

        for sensor_name, info in sensor_data.items():
            if not isinstance(info, dict):
                continue

            # Try delta_pct first
            delta_pct = info.get("delta_pct")
            if delta_pct is not None:
                raw_weights[sensor_name] = abs(float(delta_pct))
                continue

            # Compute from value and baseline
            value = info.get("value")
            baseline = info.get("baseline")

            if value is not None and baseline is not None and baseline != 0:
                deviation = abs(float(value) - float(baseline)) / abs(float(baseline))
                raw_weights[sensor_name] = deviation
            elif value is not None:
                # No baseline, use absolute value as proxy
                raw_weights[sensor_name] = abs(float(value))

        # If no weights computed, use equal distribution
        if not raw_weights:
            n = len(sensor_data)
            if n == 0:
                return []
            equal_weight = 1.0 / n
            return [
                SensorContribution(sensor_name=name, weight=equal_weight)
                for name in sensor_data
            ]

        # Normalize to sum = 1.0
        total = sum(raw_weights.values())
        if total == 0:
            n = len(raw_weights)
            equal_weight = 1.0 / n if n > 0 else 0.0
            return [
                SensorContribution(sensor_name=name, weight=equal_weight)
                for name in raw_weights
            ]

        return [
            SensorContribution(sensor_name=name, weight=w / total)
            for name, w in raw_weights.items()
        ]
