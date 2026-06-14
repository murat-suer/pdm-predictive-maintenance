"""
src/ml/fault_classifier.py
=====================================
Fault Type Classifier system that identifies specific fault types
(Bearing/Oil/Fouling/Belt) from anomaly data using SHAP features and sensor patterns.

Supports AC (Air Compressor), HX (Heat Exchanger), and CM (Conveyor Motor) machines.

This is a rule-based signature library (weighted sensor-threshold rules per
machine type), not a learned classifier. It deliberately ABSTAINS — returning
UNCLASSIFIED_ANOMALY — when no signature clears the confidence threshold, which
is common for ambiguous or early-stage degradation. Abstaining is intentional:
a recurring unidentified anomaly is escalated to a technician inspection by the
decision layer (see decision/observation_policy.py) rather than mislabelled.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from src.database.models import MachineBaseline


@dataclass
class FaultClassification:
    """Result of fault classification for an anomaly."""

    fault_type: str
    fault_description: str | dict[str, Any]
    fault_confidence: float
    matched_rules: list[str]
    shap_contributors: dict[str, float]


# Threshold for classification confidence (40%)
CONFIDENCE_THRESHOLD = 0.40

# Fault type constants
UNCLASSIFIED_ANOMALY = "UNCLASSIFIED_ANOMALY"

# AC Machine Faults
BEARING_FAULT = "BEARING_FAULT"
OIL_DEGRADATION = "OIL_DEGRADATION"
VALVE_LEAK = "VALVE_LEAK"
MOTOR_OVERLOAD = "MOTOR_OVERLOAD"

# HX Machine Faults
FOULING = "FOULING"
FLOW_RESTRICTION = "FLOW_RESTRICTION"

# CM Machine Faults
BELT_SLIP = "BELT_SLIP"


# AC Machine Fault Rules
# Each rule defines sensor patterns and SHAP contributors for specific faults
AC_FAULT_RULES = [
    {
        "fault_type": BEARING_FAULT,
        "description": {"key": "fault_ac_bearing_desc"},
        "sensor_patterns": {
            "vibration_rms": {"min": 4.5, "weight": 0.35},
            "bearing_temp": {"min": 85.0, "weight": 0.30},
        },
        "shap_sensors": ["vibration_rms", "bearing_temp"],
        "base_confidence": 0.55,
    },
    {
        "fault_type": OIL_DEGRADATION,
        "description": {"key": "fault_ac_oil_desc"},
        "sensor_patterns": {
            "oil_pressure": {"max": 2.8, "weight": 0.20},
            "bearing_temp": {"min": 75.0, "weight": 0.30},
        },
        "shap_sensors": ["oil_pressure", "bearing_temp"],
        "base_confidence": 0.50,
    },
    {
        "fault_type": VALVE_LEAK,
        "description": {"key": "fault_ac_valve_desc"},
        "sensor_patterns": {
            "outlet_pressure": {"max": 6.5, "weight": 0.05},
            "motor_current": {"min": 24.0, "weight": 0.10},
        },
        "shap_sensors": ["outlet_pressure", "motor_current"],
        "base_confidence": 0.45,
    },
    {
        "fault_type": MOTOR_OVERLOAD,
        "description": {"key": "fault_ac_motor_desc"},
        "sensor_patterns": {
            "motor_current": {"min": 28.0, "weight": 0.10},
            "vibration_rms": {"min": 3.5, "weight": 0.35},
        },
        "shap_sensors": ["motor_current", "vibration_rms"],
        "base_confidence": 0.48,
    },
]


# HX Machine Fault Rules
HX_FAULT_RULES = [
    {
        "fault_type": FOULING,
        "description": {"key": "fault_hx_fouling_desc"},
        "sensor_patterns": {
            "fouling_index": {"min": 0.35, "weight": 0.10},
            "pressure_drop": {"min": 1.4, "weight": 0.35},
            "outlet_temp": {"min": 95.0, "weight": 0.25},
        },
        "shap_sensors": ["fouling_index", "pressure_drop", "outlet_temp"],
        "base_confidence": 0.52,
    },
    {
        "fault_type": FLOW_RESTRICTION,
        "description": {"key": "fault_hx_flow_desc"},
        "sensor_patterns": {
            "flow_rate": {"max": 9.0, "weight": 0.15},
            "pressure_drop": {"min": 1.2, "weight": 0.35},
        },
        "shap_sensors": ["flow_rate", "pressure_drop"],
        "base_confidence": 0.48,
    },
]


# CM Machine Fault Rules
CM_FAULT_RULES = [
    {
        "fault_type": BELT_SLIP,
        "description": {"key": "fault_cm_belt_desc"},
        "sensor_patterns": {
            "belt_tension": {"min": 11.5, "weight": 0.30},
            "speed_rpm": {"max": 1380.0, "weight": 0.15},
        },
        "shap_sensors": ["belt_tension", "speed_rpm"],
        "base_confidence": 0.50,
    },
    {
        "fault_type": MOTOR_OVERLOAD,
        "description": {"key": "fault_cm_motor_desc"},
        "sensor_patterns": {
            "motor_load": {"min": 88.0, "weight": 0.20},
            "drive_temp": {"min": 78.0, "weight": 0.25},
        },
        "shap_sensors": ["motor_load", "drive_temp"],
        "base_confidence": 0.48,
    },
    {
        "fault_type": BEARING_FAULT,
        "description": {"key": "fault_cm_bearing_desc"},
        "sensor_patterns": {
            "vibration_rms": {"min": 3.5, "weight": 0.10},
            "drive_temp": {"min": 70.0, "weight": 0.25},
        },
        "shap_sensors": ["vibration_rms", "drive_temp"],
        "base_confidence": 0.45,
    },
]


# Machine type to rules mapping
MACHINE_RULES = {
    "AC": AC_FAULT_RULES,
    "HX": HX_FAULT_RULES,
    "CM": CM_FAULT_RULES,
}


class FaultClassifier:
    """
    Classifies anomalies into specific fault types based on sensor patterns
    and SHAP feature contributions.
    """

    def __init__(self, db_session_factory: Callable[[], Session] | None = None):
        """Initialize the fault classifier with rule sets."""
        self.ac_rules = AC_FAULT_RULES
        self.hx_rules = HX_FAULT_RULES
        self.cm_rules = CM_FAULT_RULES
        self.machine_rules = MACHINE_RULES
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.db_session_factory = db_session_factory
        self._baseline_cache: dict[str, dict[str, tuple[float, float]]] = {}

    def classify(
        self,
        machine_id: str,
        machine_type: str,
        anomaly_score: float,
        shap_values: dict[str, float] | None = None,
        sensor_readings: dict[str, float] | None = None,
        top_contributing_sensor: str | None = None,
    ) -> FaultClassification:
        """
        Classify an anomaly into a specific fault type.

        Args:
            machine_id: Unique identifier for the machine
            machine_type: Type of machine (AC, HX, CM)
            anomaly_score: The anomaly detection score
            shap_values: Dictionary of SHAP feature contributions
            sensor_readings: Current sensor readings
            top_contributing_sensor: Sensor with highest SHAP value

        Returns:
            FaultClassification with fault type, description, and confidence
        """
        shap_values = shap_values or {}
        sensor_readings = sensor_readings or {}

        # Get rules for this machine type
        rules = self.machine_rules.get(machine_type, [])
        if not rules:
            return FaultClassification(
                fault_type=UNCLASSIFIED_ANOMALY,
                fault_description={"key": "fault_unclassified_short"},
                fault_confidence=0.0,
                matched_rules=[],
                shap_contributors=shap_values,
            )

        # Evaluate all rules and find the best match
        rule_scores = []
        for rule in rules:
            score, matched = self._evaluate_rule(
                rule,
                machine_id,
                shap_values,
                sensor_readings,
                top_contributing_sensor,
            )
            rule_scores.append(
                {
                    "rule": rule,
                    "score": score,
                    "matched": matched,
                }
            )

        # Sort by score descending
        rule_scores.sort(key=lambda x: x["score"], reverse=True)

        matched_scores = [candidate for candidate in rule_scores if candidate["matched"]]

        # Get the best matching rule
        if matched_scores and matched_scores[0]["score"] >= self.confidence_threshold:
            best = matched_scores[0]
            return FaultClassification(
                fault_type=best["rule"]["fault_type"],
                fault_description=best["rule"]["description"],
                fault_confidence=best["score"],
                matched_rules=[best["rule"]["fault_type"]],
                shap_contributors=shap_values,
            )

        # Return unclassified if no rule meets threshold
        return FaultClassification(
            fault_type=UNCLASSIFIED_ANOMALY,
            fault_description={"key": "fault_unclassified_short"},
            fault_confidence=rule_scores[0]["score"] if rule_scores else 0.0,
            matched_rules=[r["rule"]["fault_type"] for r in rule_scores if r["matched"]],
            shap_contributors=shap_values,
        )

    def _evaluate_rule(
        self,
        rule: dict[str, Any],
        machine_id: str,
        shap_values: dict[str, float],
        sensor_readings: dict[str, float],
        top_contributing_sensor: str | None,
    ) -> tuple:
        """
        Evaluate a single fault rule against the current data.

        Args:
            rule: Fault rule dictionary
            shap_values: SHAP feature contributions
            sensor_readings: Current sensor readings
            top_contributing_sensor: Sensor with highest SHAP value

        Returns:
            Tuple of (confidence_score, matched_boolean)
        """
        score = rule["base_confidence"]
        matched = False

        # Check sensor patterns
        sensor_patterns = rule.get("sensor_patterns", {})
        pattern_matches = 0

        for sensor, condition in sensor_patterns.items():
            reading = sensor_readings.get(sensor)
            if reading is None:
                continue

            match = self._matches_sensor_pattern(machine_id, sensor, reading, condition)

            if match:
                pattern_matches += 1
                # Boost score based on sensor weight
                score += condition.get("weight", 0.1) * 0.2

        # Check SHAP contributions
        shap_sensors = rule.get("shap_sensors", [])
        shap_boost = 0.0

        for sensor in shap_sensors:
            shap_val = shap_values.get(f"{sensor}_value", shap_values.get(sensor, 0.0))
            if shap_val > 0:
                shap_boost += shap_val * 0.15

        score += shap_boost

        # Bonus if top contributing sensor matches rule sensors
        if top_contributing_sensor and top_contributing_sensor in shap_sensors:
            score += 0.10

        # Mark as matched if at least one sensor pattern matches
        if pattern_matches > 0:
            matched = True

        # If nothing matched, reduce confidence significantly
        if not matched:
            score = score * 0.5  # Reduce by 50% if no patterns matched

        # Cap confidence at 0.95
        score = min(score, 0.95)

        return score, matched

    def _get_baseline(self, machine_id: str, sensor: str) -> tuple[float, float] | None:
        """Return active healthy baseline mean and std for one sensor."""
        if self.db_session_factory is None:
            return None

        if machine_id not in self._baseline_cache:
            with self.db_session_factory() as session:
                rows = (
                    session.query(MachineBaseline)
                    .filter(
                        MachineBaseline.machine_id == machine_id,
                        MachineBaseline.is_active == True,
                    )
                    .all()
                )
            self._baseline_cache[machine_id] = {row.sensor: (row.mean_value, row.std_value) for row in rows}

        return self._baseline_cache[machine_id].get(sensor)

    def _matches_sensor_pattern(
        self,
        machine_id: str,
        sensor: str,
        reading: float,
        condition: dict[str, Any],
    ) -> bool:
        """Evaluate one directional rule using baseline z-score when available."""
        baseline = self._get_baseline(machine_id, sensor)
        if baseline:
            mean, std = baseline
            z_score = (reading - mean) / max(std, 1e-6)
            k_sigma = condition.get("k_sigma", 4.0)
            if "min" in condition and z_score >= k_sigma:
                return True
            if "max" in condition and z_score <= -k_sigma:
                return True
            return False

        return bool(
            ("min" in condition and reading >= condition["min"]) or ("max" in condition and reading <= condition["max"])
        )

    def _top_shap_sensor(self, shap_values: dict[str, float]) -> str | None:
        """
        Find the sensor with the highest absolute SHAP contribution.

        Args:
            shap_values: Dictionary of SHAP feature contributions

        Returns:
            Name of the top contributing sensor or None
        """
        if not shap_values:
            return None

        return max(shap_values.items(), key=lambda x: abs(x[1]))[0]

    def get_supported_fault_types(self, machine_type: str) -> list[str]:
        """
        Get list of supported fault types for a machine type.

        Args:
            machine_type: Type of machine (AC, HX, CM)

        Returns:
            List of fault type strings
        """
        rules = self.machine_rules.get(machine_type, [])
        return [rule["fault_type"] for rule in rules]

    def get_fault_description(self, fault_type: str) -> dict:
        """
        Get i18n key object for a fault type.

        Args:
            fault_type: Fault type constant

        Returns:
            Dict with i18n key
        """
        descriptions = {
            BEARING_FAULT: {"key": "fault_bearing_short"},
            OIL_DEGRADATION: {"key": "fault_oil_short"},
            VALVE_LEAK: {"key": "fault_valve_short"},
            MOTOR_OVERLOAD: {"key": "fault_motor_short"},
            FOULING: {"key": "fault_fouling_short"},
            FLOW_RESTRICTION: {"key": "fault_flow_restriction_short"},
            BELT_SLIP: {"key": "fault_belt_slip_short"},
            UNCLASSIFIED_ANOMALY: {"key": "fault_unclassified_short"},
        }
        return descriptions.get(fault_type, {"key": "fault_unknown"})
