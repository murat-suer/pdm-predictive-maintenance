"""
src/ml/mhi_calculator.py
==================================
Machine Health Index (MHI) computation.
MHI = Availability × Reliability × Condition

CRITICAL: "OEE" is BANNED. This is MHI — no production counters required.
API endpoint: GET /api/v1/machines/{id}/mhi
Dashboard label: "MHI: 87% (Good)"
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def reliability_score(
    anomaly_rate: float,
    baseline_rate: float = 0.05,
    tolerance_band: float = 0.15,
) -> float:
    """Reliability degrades linearly above baseline anomaly rate.

    At anomaly_rate == baseline_rate: reliability = 1.0
    At anomaly_rate == baseline_rate + tolerance_band: reliability = 0.0

    Args:
        anomaly_rate: Measured fraction of anomalous readings (0..1).
        baseline_rate: Expected contamination rate (default 0.05 = IF contamination).
        tolerance_band: Width above baseline before reliability hits zero (default 0.15).
    """
    excess = max(0.0, anomaly_rate - baseline_rate)
    return max(0.0, 1.0 - excess / tolerance_band)


class MHICalculator:
    """
    Computes Machine Health Index from anomaly detector outputs and phase data.

    MHI = Availability × Reliability × Condition
    - Availability: fraction of time machine is running (not FAILED/IDLE)
    - Reliability:  inverse of anomaly frequency
    - Condition:    current degradation state (1 - degradation_level)

    EMA smoothing is applied to prevent sudden MHI drops caused by abrupt
    degradation changes (e.g., chaos injection events).

    Result range: 0.0–1.0
    Classification:
        Excellent       : >= 0.85
        Good            : >= 0.70
        Degrading       : >= 0.55
        Critical        : < 0.55
    """

    CLASSIFICATION_THRESHOLDS = [
        (0.85, "Excellent"),
        (0.70, "Good"),
        (0.55, "Degrading"),
        (0.00, "Critical — Action Required"),
    ]

    DEFAULT_EMA_ALPHA = 0.3
    # Below this many readings the anomaly rate is statistically meaningless
    # (2 anomalies in 9 readings = "0% reliability" on a machine fresh out of
    # maintenance). Reliability is reported as None and treated as neutral.
    MIN_RELIABILITY_SAMPLES = 20
    # Event-based reliability: each CONFIRMED anomaly event in the window
    # costs this fraction. Raw detector flags are too jumpy for a headline
    # metric — an IsolationForest grumbling at score≈0.5 produced "0%
    # reliability" on machines whose alarm pipeline (grace + 2-cycle
    # confirmation) was completely quiet.
    RELIABILITY_PENALTY_PER_EVENT = 0.10
    # Nothing is ever 100% reliable: a quiet history is evidence, not a
    # guarantee. The metric is capped just like RUL confidence.
    RELIABILITY_CEILING = 0.99

    def __init__(self, ema_alpha: float | None = None):
        self.ema_alpha = ema_alpha if ema_alpha is not None else self.DEFAULT_EMA_ALPHA
        self._ema_cache: dict[str, float] = {}

    def compute(
        self,
        machine_id: str,
        phase: str,
        degradation_level: float,
        recent_anomaly_count: int,
        recent_readings_count: int,
        recent_downtime_minutes: float = 0.0,
        window_hours: float = 24.0,
        confirmed_events: int | None = None,
    ) -> dict:
        """
        Compute MHI for a machine.

        Args:
            machine_id: Machine identifier (e.g. "AC-201")
            phase: Current machine phase
            degradation_level: Current degradation (0.0 → 1.0)
            recent_anomaly_count: Anomaly events in window
            recent_readings_count: Total readings in window
            recent_downtime_minutes: Downtime in window
            window_hours: Analysis window in hours

        Returns:
            {
                machine_id: str,
                calculated_at: datetime,
                health_score: float,
                availability_score: float,
                reliability_score: float,
                condition_score: float,
                classification: str,
            }
        """
        window_minutes = window_hours * 60
        availability = max(0.0, min(1.0, (window_minutes - recent_downtime_minutes) / window_minutes))
        if phase in ("FAILED", "IDLE"):
            availability = max(0.0, availability - 0.15)

        if confirmed_events is not None:
            # Preferred: ground reliability in the same evidence the alarm
            # system acts on. Each confirmed anomaly event in the window
            # docks a fixed fraction.
            reliability = max(
                0.0, 1.0 - self.RELIABILITY_PENALTY_PER_EVENT * confirmed_events
            )
        elif recent_readings_count >= self.MIN_RELIABILITY_SAMPLES:
            anomaly_rate = min(1.0, recent_anomaly_count / recent_readings_count)
            reliability = reliability_score(anomaly_rate)
        else:
            # Not enough evidence — neutral in the product, None for display.
            reliability = None

        if reliability is not None:
            reliability = min(reliability, self.RELIABILITY_CEILING)

        condition = max(0.0, 1.0 - degradation_level)

        mhi = availability * (reliability if reliability is not None else 1.0) * condition

        if phase == "FAILED":
            mhi = 0.0
        elif phase == "ANOMALY":
            mhi = min(mhi, 0.60)

        if machine_id in self._ema_cache:
            prev = self._ema_cache[machine_id]
            mhi = self.ema_alpha * mhi + (1.0 - self.ema_alpha) * prev
        self._ema_cache[machine_id] = mhi

        mhi = round(max(0.0, min(1.0, mhi)), 4)
        availability = round(availability, 4)
        reliability = round(reliability, 4) if reliability is not None else None
        condition = round(condition, 4)

        classification = self._classify(mhi)

        return {
            "machine_id": machine_id,
            "calculated_at": datetime.now(UTC),
            "health_score": mhi,
            "availability_score": availability,
            "reliability_score": reliability,
            "condition_score": condition,
            "classification": classification,
            "rul_hours": None,
            "confidence": None,
        }

    def _classify(self, mhi: float) -> str:
        for threshold, label in self.CLASSIFICATION_THRESHOLDS:
            if mhi >= threshold:
                return label
        return "Critical — Action Required"
