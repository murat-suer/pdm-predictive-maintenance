"""
YAML-based FMEA failure mode matching library.

Loads failure modes from YAML, evaluates sensor signatures against rules,
calculates confidence scores, and ranks results by confidence * RPN.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SignatureRule:
    """A single sensor signature rule."""
    sensor: str
    condition: str


@dataclass
class FailureMode:
    """A failure mode with FMEA scoring and sensor signature."""
    mode_id: str
    category: str
    description: str
    rpn: int
    rules: list[SignatureRule]
    category_weight: float = 0.8


@dataclass
class MatchResult:
    """Result of matching sensor data against a failure mode."""
    mode_id: str
    confidence: float
    rpn: int
    category: str = ""
    description: str = ""


# Default YAML path (bundled with module)
_DEFAULT_YAML = Path(__file__).parent / "failure_modes.yaml"


class FailureModeLibrary:
    """
    YAML-based failure mode matching library.

    Loads failure modes from YAML, evaluates sensor signatures,
    and returns ranked match results.
    """

    def __init__(self, yaml_path: str | None = None):
        if yaml_path is not None:
            path = Path(yaml_path)
        else:
            path = _DEFAULT_YAML

        if not path.exists():
            raise FileNotFoundError(f"Failure modes YAML not found: {path}")

        self._modes: list[FailureMode] = []
        self._load_yaml(path)

    def _load_yaml(self, path: Path):
        """Load and parse the YAML file."""
        with open(path, encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Malformed YAML: {e}") from e

        if not data:
            raise ValueError("Empty YAML file or no content.")

        if not isinstance(data, dict) or "failure_modes" not in data:
            raise KeyError("YAML must contain 'failure_modes' key.")

        modes_data = data["failure_modes"]
        if not modes_data or not isinstance(modes_data, list):
            raise ValueError("'failure_modes' must be a non-empty list.")

        seen_ids = set()
        for mode_dict in modes_data:
            mode_id = mode_dict.get("mode_id")
            if not mode_id:
                raise KeyError("Each failure mode must have a 'mode_id'.")

            if mode_id in seen_ids:
                raise ValueError(f"Duplicate mode_id: '{mode_id}'")
            seen_ids.add(mode_id)

            rules = []
            for rule_dict in mode_dict.get("rules", []):
                rules.append(SignatureRule(
                    sensor=rule_dict["sensor"],
                    condition=rule_dict["condition"],
                ))

            mode = FailureMode(
                mode_id=mode_id,
                category=mode_dict.get("category", ""),
                description=mode_dict.get("description", ""),
                rpn=int(mode_dict.get("rpn", 0)),
                rules=rules,
                category_weight=float(mode_dict.get("category_weight", 0.8)),
            )
            self._modes.append(mode)

    def get_all_modes(self) -> list[FailureMode]:
        """Return all loaded failure modes."""
        return list(self._modes)

    def _get_sensor_value(self, sensor_data: dict, sensor_name: str) -> dict | None:
        """
        Extract sensor value info from data.

        Handles both formats:
        - Simple: {"sensor": 5.5} → {"value": 5.5}
        - Dict: {"sensor": {"value": 5.5, "trend": "up"}}
        """
        if sensor_name not in sensor_data:
            return None

        raw = sensor_data[sensor_name]
        if isinstance(raw, (int, float)):
            return {"value": float(raw)}
        elif isinstance(raw, dict):
            return raw
        return None

    def _evaluate_rule(self, rule: SignatureRule, sensor_data: dict) -> bool:
        """
        Evaluate a single rule against sensor data.

        Rule types:
        - trend_up: sensor trend is "up"
        - trend_down: sensor trend is "down"
        - above:X: value > X
        - below:X: value < X
        - delta_pct:X: delta_pct > X
        """
        info = self._get_sensor_value(sensor_data, rule.sensor)
        if info is None:
            return False

        condition = rule.condition.strip()

        if condition == "trend_up":
            return info.get("trend", "").lower() == "up"

        elif condition == "trend_down":
            return info.get("trend", "").lower() == "down"

        elif condition.startswith("above:"):
            threshold = float(condition.split(":")[1])
            value = info.get("value")
            if value is None:
                return False
            return value > threshold

        elif condition.startswith("below:"):
            threshold = float(condition.split(":")[1])
            value = info.get("value")
            if value is None:
                return False
            return value < threshold

        elif condition.startswith("delta_pct:"):
            threshold = float(condition.split(":")[1])
            delta = info.get("delta_pct")
            if delta is None:
                return False
            return float(delta) > threshold

        return False

    def match(
        self,
        sensor_data: dict,
        machine_id: str,
        min_confidence: float = 0.0,
    ) -> list[MatchResult]:
        """
        Match sensor data against all failure modes.

        Returns ranked results (confidence * RPN, descending).

        Args:
            sensor_data: Sensor readings (dict format)
            machine_id: Machine identifier
            min_confidence: Minimum confidence threshold (0.0-1.0)
        """
        if sensor_data is None:
            raise TypeError("sensor_data cannot be None")

        if not sensor_data:
            return []

        results = []

        for mode in self._modes:
            if not mode.rules:
                continue

            matched_count = 0
            total_rules = len(mode.rules)

            for rule in mode.rules:
                if self._evaluate_rule(rule, sensor_data):
                    matched_count += 1

            # Confidence = matched/total * category_weight
            confidence = (matched_count / total_rules) * mode.category_weight

            if confidence >= min_confidence and confidence > 0:
                results.append(MatchResult(
                    mode_id=mode.mode_id,
                    confidence=confidence,
                    rpn=mode.rpn,
                    category=mode.category,
                    description=mode.description,
                ))

        # Rank by confidence * RPN (descending)
        results.sort(key=lambda r: r.confidence * r.rpn, reverse=True)

        return results
