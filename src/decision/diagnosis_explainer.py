"""
Diagnosis Explainer (Phase 2E - UX/Demo Layer).

Reliability scoring and sensor contribution explanation for
diagnostic results.

KRITIK FORMULAS:
  1. reliability = 0.6 * signature_confidence + 0.4 * ml_score
  2. sum(sensor_contributions) == 1.0
  3. signature_sensor_weight = raw_weight * 1.5

Features:
  - Weighted reliability calculation
  - Sensor contribution with signature boost
  - Normalization to sum=1.0
  - Missing baseline/std handling
  - Negative/zero value handling
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RELIABILITY_SIGNATURE_WEIGHT: float = 0.6
RELIABILITY_ML_WEIGHT: float = 0.4
SIGNATURE_SENSOR_BOOST: float = 1.5
MIN_STD_DEV: float = 1e-6


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ReliabilityScore:
    """Reliability score breakdown."""
    reliability: float
    signature_confidence: float
    ml_score: float


@dataclass
class SensorContribution:
    """Single sensor's contribution to the diagnosis."""
    sensor_name: str
    weight: float
    raw_deviation: float = 0.0
    is_signature: bool = False


@dataclass
class ExplanationResult:
    """Full explanation result combining reliability and contributions."""
    reliability: float
    sensor_contributions: list[SensorContribution]
    reliability_score: ReliabilityScore | None = None


# ---------------------------------------------------------------------------
# DiagnosisExplainer
# ---------------------------------------------------------------------------
class DiagnosisExplainer:
    """
    Computes reliability scores and sensor contributions for diagnostics.

    reliability = 0.6 * signature_confidence + 0.4 * ml_score
    sensor_contribution = |current - baseline| / std (normalized, with 1.5x boost for signature sensors)
    """

    def calculate_reliability(
        self, signature_confidence: float, ml_score: float
    ) -> float:
        """
        Calculate reliability as weighted combination.

        reliability = 0.6 * signature_confidence + 0.4 * ml_score
        """
        return (
            RELIABILITY_SIGNATURE_WEIGHT * signature_confidence
            + RELIABILITY_ML_WEIGHT * ml_score
        )

    def compute_contributions(
        self,
        sensor_data: dict[str, dict],
        signature_sensors: list[str],
    ) -> list[SensorContribution]:
        """
        Compute normalized sensor contributions.

        For each sensor:
          raw_weight = |current - baseline| / max(std, MIN_STD_DEV)
          if sensor is signature: raw_weight *= SIGNATURE_SENSOR_BOOST

        Then normalize so sum(weights) == 1.0.
        If all weights are zero, distribute equally.
        """
        if not sensor_data:
            return []

        signature_set = set(signature_sensors)
        contributions: list[SensorContribution] = []

        for sensor_name, data in sensor_data.items():
            current = data.get("current", 0.0)
            baseline = data.get("baseline")
            std = data.get("std")
            is_sig = data.get("is_signature", False) or (sensor_name in signature_set)

            # Handle missing baseline: use current as fallback (deviation = 0)
            if baseline is None:
                baseline = current

            # Handle missing/zero/negative std
            if std is None or std <= 0:
                std = MIN_STD_DEV
            else:
                std = max(abs(std), MIN_STD_DEV)

            # Compute raw deviation
            raw_deviation = abs(current - baseline) / std

            # Apply signature boost
            weight = raw_deviation
            if is_sig:
                weight *= SIGNATURE_SENSOR_BOOST

            contributions.append(SensorContribution(
                sensor_name=sensor_name,
                weight=weight,
                raw_deviation=raw_deviation,
                is_signature=is_sig,
            ))

        # Normalize weights to sum to 1.0
        total = sum(c.weight for c in contributions)
        if total > 0:
            for c in contributions:
                c.weight = c.weight / total
        else:
            # Equal distribution when all deviations are zero
            n = len(contributions)
            if n > 0:
                equal_weight = 1.0 / n
                for c in contributions:
                    c.weight = equal_weight

        return contributions

    def explain(
        self,
        signature_confidence: float,
        ml_score: float,
        sensor_data: dict[str, dict],
        signature_sensors: list[str],
    ) -> ExplanationResult:
        """
        Full explanation: reliability + sensor contributions.
        """
        reliability = self.calculate_reliability(signature_confidence, ml_score)
        contributions = self.compute_contributions(sensor_data, signature_sensors)

        score = ReliabilityScore(
            reliability=reliability,
            signature_confidence=signature_confidence,
            ml_score=ml_score,
        )

        return ExplanationResult(
            reliability=reliability,
            sensor_contributions=contributions,
            reliability_score=score,
        )
