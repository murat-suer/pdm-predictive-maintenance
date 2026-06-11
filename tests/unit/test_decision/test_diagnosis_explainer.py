"""
Unit tests for src.decision.diagnosis_explainer (Phase 2E - UX/Demo Layer)

Tests DiagnosisExplainer: reliability scoring + sensor contribution explanation.
  - reliability = 0.6 * signature_confidence + 0.4 * ml_score
  - Sensor contribution: |current - baseline| / std, normalize to sum=1.0
  - Signature sensors get 1.5x weight boost
  - Missing baseline handling
  - Negative/zero values handling
  - Weight validation (must sum > 0)

KRITIK FORMULAS:
  1. reliability = 0.6 * signature_confidence + 0.4 * ml_score
  2. sum(sensor_contributions) == 1.0
  3. signature_sensor_weight = raw_weight * 1.5

NOTE: These tests will FAIL until diagnosis_explainer.py is implemented.
"""


import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.diagnosis_explainer import (
    MIN_STD_DEV,
    RELIABILITY_ML_WEIGHT,
    RELIABILITY_SIGNATURE_WEIGHT,
    SIGNATURE_SENSOR_BOOST,
    DiagnosisExplainer,
    ReliabilityScore,
)


# ---------------------------------------------------------------------------
# Constants verification
# ---------------------------------------------------------------------------
class TestConstants:
    """Verify module-level constants."""

    def test_reliability_signature_weight(self):
        """Signature weight must be 0.6."""
        assert RELIABILITY_SIGNATURE_WEIGHT == 0.6

    def test_reliability_ml_weight(self):
        """ML weight must be 0.4."""
        assert RELIABILITY_ML_WEIGHT == 0.4

    def test_weights_sum_to_one(self):
        """Reliability weights must sum to 1.0."""
        assert abs(RELIABILITY_SIGNATURE_WEIGHT + RELIABILITY_ML_WEIGHT - 1.0) < 1e-9

    def test_signature_sensor_boost(self):
        """Signature sensor boost must be 1.5x."""
        assert SIGNATURE_SENSOR_BOOST == 1.5

    def test_min_std_dev_positive(self):
        """Minimum std dev must be positive (prevents division by zero)."""
        assert MIN_STD_DEV > 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def explainer():
    """Fresh DiagnosisExplainer."""
    return DiagnosisExplainer()


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data with baselines and std devs."""
    return {
        "vibration_rms": {
            "current": 8.5,
            "baseline": 4.0,
            "std": 1.2,
            "is_signature": True,
        },
        "temperature": {
            "current": 75.0,
            "baseline": 60.0,
            "std": 5.0,
            "is_signature": False,
        },
        "pressure": {
            "current": 3.2,
            "baseline": 3.0,
            "std": 0.3,
            "is_signature": False,
        },
    }


@pytest.fixture
def signature_sensors():
    """List of signature sensor names."""
    return ["vibration_rms"]


# ---------------------------------------------------------------------------
# TestReliabilityFormula - KRITIK
# ---------------------------------------------------------------------------
class TestReliabilityFormula:
    """
    KRITIK: reliability = 0.6 * signature_confidence + 0.4 * ml_score
    This formula must be exact.
    """

    def test_basic_formula(self, explainer):
        """Basic reliability calculation: 0.6 * 0.8 + 0.4 * 0.7 = 0.76."""
        result = explainer.calculate_reliability(0.8, 0.7)
        expected = 0.6 * 0.8 + 0.4 * 0.7  # = 0.48 + 0.28 = 0.76
        assert abs(result - expected) < 1e-9

    def test_all_signature_no_ml(self, explainer):
        """signature=1.0, ml=0.0 → reliability = 0.6."""
        result = explainer.calculate_reliability(1.0, 0.0)
        assert abs(result - 0.6) < 1e-9

    def test_all_ml_no_signature(self, explainer):
        """signature=0.0, ml=1.0 → reliability = 0.4."""
        result = explainer.calculate_reliability(0.0, 1.0)
        assert abs(result - 0.4) < 1e-9

    def test_both_zero(self, explainer):
        """signature=0.0, ml=0.0 → reliability = 0.0."""
        result = explainer.calculate_reliability(0.0, 0.0)
        assert abs(result - 0.0) < 1e-9

    def test_both_one(self, explainer):
        """signature=1.0, ml=1.0 → reliability = 1.0."""
        result = explainer.calculate_reliability(1.0, 1.0)
        assert abs(result - 1.0) < 1e-9

    def test_mid_values(self, explainer):
        """signature=0.5, ml=0.5 → reliability = 0.5."""
        result = explainer.calculate_reliability(0.5, 0.5)
        assert abs(result - 0.5) < 1e-9

    def test_signature_dominant(self, explainer):
        """High signature, low ML → result closer to signature."""
        result = explainer.calculate_reliability(0.9, 0.1)
        expected = 0.6 * 0.9 + 0.4 * 0.1  # = 0.54 + 0.04 = 0.58
        assert abs(result - expected) < 1e-9

    def test_ml_dominant(self, explainer):
        """Low signature, high ML → result closer to ML contribution."""
        result = explainer.calculate_reliability(0.1, 0.9)
        expected = 0.6 * 0.1 + 0.4 * 0.9  # = 0.06 + 0.36 = 0.42
        assert abs(result - expected) < 1e-9

    def test_reliability_bounded_0_1(self, explainer):
        """Reliability must be in [0, 1] for valid inputs."""
        for sig in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for ml in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = explainer.calculate_reliability(sig, ml)
                assert 0.0 <= result <= 1.0, f"sig={sig}, ml={ml}, rel={result}"

    def test_reliability_is_linear_combination(self, explainer):
        """Verify linearity: R(a*s1 + b*s2, a*m1 + b*m2) = a*R(s1,m1) + b*R(s2,m2)
        when a+b=1."""
        # R(0.5*1.0 + 0.5*0.0, 0.5*1.0 + 0.5*0.0) should equal 0.5*R(1,1) + 0.5*R(0,0)
        r_mid = explainer.calculate_reliability(0.5, 0.5)
        r_high = explainer.calculate_reliability(1.0, 1.0)
        r_low = explainer.calculate_reliability(0.0, 0.0)
        expected = 0.5 * r_high + 0.5 * r_low
        assert abs(r_mid - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestSensorContribution
# ---------------------------------------------------------------------------
class TestSensorContribution:
    """
    Sensor contribution: |current - baseline| / std, then normalize.
    Signature sensors get 1.5x weight boost.
    """

    def test_contributions_sum_to_one(self, explainer, sample_sensor_data, signature_sensors):
        """Sensor contributions must sum to 1.0."""
        contributions = explainer.compute_contributions(sample_sensor_data, signature_sensors)
        total = sum(c.weight for c in contributions)
        assert abs(total - 1.0) < 1e-6

    def test_contribution_count_matches_sensors(self, explainer, sample_sensor_data, signature_sensors):
        """Number of contributions equals number of sensors."""
        contributions = explainer.compute_contributions(sample_sensor_data, signature_sensors)
        assert len(contributions) == len(sample_sensor_data)

    def test_signature_sensor_gets_boost(self, explainer, signature_sensors):
        """Signature sensor gets 1.5x weight boost."""
        # Create data where vibration_rms has same raw deviation as temperature
        sensor_data = {
            "vibration_rms": {
                "current": 6.0,
                "baseline": 4.0,
                "std": 1.0,
                "is_signature": True,
            },
            "temperature": {
                "current": 8.0,
                "baseline": 6.0,
                "std": 1.0,
                "is_signature": False,
            },
        }
        # Raw: both have |deviation|/std = 2.0
        # After boost: vibration_rms = 2.0 * 1.5 = 3.0, temperature = 2.0
        # Normalized: vibration_rms = 3.0/5.0 = 0.6, temperature = 2.0/5.0 = 0.4
        contributions = explainer.compute_contributions(sensor_data, signature_sensors)

        vib = next(c for c in contributions if c.sensor_name == "vibration_rms")
        temp = next(c for c in contributions if c.sensor_name == "temperature")

        # Signature sensor should have higher weight
        assert vib.weight > temp.weight
        # Ratio should be 1.5 (boost factor)
        ratio = vib.weight / temp.weight
        assert abs(ratio - 1.5) < 1e-6

    def test_larger_deviation_higher_weight(self, explainer):
        """Sensor with larger |current - baseline| / std gets higher weight."""
        sensor_data = {
            "sensor_a": {"current": 10.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
            "sensor_b": {"current": 6.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
        }
        # sensor_a: |10-5|/1 = 5.0, sensor_b: |6-5|/1 = 1.0
        contributions = explainer.compute_contributions(sensor_data, [])
        a = next(c for c in contributions if c.sensor_name == "sensor_a")
        b = next(c for c in contributions if c.sensor_name == "sensor_b")
        assert a.weight > b.weight

    def test_zero_deviation_gets_zero_weight(self, explainer):
        """Sensor with current == baseline gets zero (or near-zero) weight."""
        sensor_data = {
            "sensor_a": {"current": 5.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
            "sensor_b": {"current": 8.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        a = next(c for c in contributions if c.sensor_name == "sensor_a")
        b = next(c for c in contributions if c.sensor_name == "sensor_b")
        assert a.weight < b.weight

    def test_single_sensor_gets_full_weight(self, explainer):
        """Single sensor gets weight = 1.0."""
        sensor_data = {
            "only_sensor": {"current": 10.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        assert len(contributions) == 1
        assert abs(contributions[0].weight - 1.0) < 1e-6

    def test_all_zero_deviations_equal_distribution(self, explainer):
        """When all deviations are zero, equal distribution."""
        sensor_data = {
            "sensor_a": {"current": 5.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
            "sensor_b": {"current": 3.0, "baseline": 3.0, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        total = sum(c.weight for c in contributions)
        assert abs(total - 1.0) < 1e-6
        # Equal distribution
        for c in contributions:
            assert abs(c.weight - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# TestMissingBaselineHandling
# ---------------------------------------------------------------------------
class TestMissingBaselineHandling:
    """Handle missing baseline gracefully."""

    def test_missing_baseline_uses_current_as_fallback(self, explainer):
        """Missing baseline should not crash; uses fallback."""
        sensor_data = {
            "sensor_a": {"current": 10.0, "std": 1.0, "is_signature": False},
            "sensor_b": {"current": 5.0, "baseline": 3.0, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        total = sum(c.weight for c in contributions)
        assert abs(total - 1.0) < 1e-6

    def test_missing_std_uses_minimum(self, explainer):
        """Missing std should use MIN_STD_DEV to prevent division by zero."""
        sensor_data = {
            "sensor_a": {"current": 10.0, "baseline": 5.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        assert len(contributions) == 1
        assert abs(contributions[0].weight - 1.0) < 1e-6

    def test_empty_sensor_data(self, explainer):
        """Empty sensor data returns empty contributions."""
        contributions = explainer.compute_contributions({}, [])
        assert len(contributions) == 0

    def test_none_baseline_treated_as_missing(self, explainer):
        """None baseline treated same as missing."""
        sensor_data = {
            "sensor_a": {"current": 10.0, "baseline": None, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        assert len(contributions) == 1


# ---------------------------------------------------------------------------
# TestNegativeZeroValues
# ---------------------------------------------------------------------------
class TestNegativeZeroValues:
    """Handle negative and zero values."""

    def test_negative_deviation_uses_absolute(self, explainer):
        """Negative deviation uses absolute value."""
        sensor_data = {
            "sensor_a": {"current": 3.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
            "sensor_b": {"current": 7.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
        }
        # Both have |deviation| = 2.0, so equal weights
        contributions = explainer.compute_contributions(sensor_data, [])
        a = next(c for c in contributions if c.sensor_name == "sensor_a")
        b = next(c for c in contributions if c.sensor_name == "sensor_b")
        assert abs(a.weight - b.weight) < 1e-6

    def test_zero_std_uses_minimum(self, explainer):
        """Zero std should use MIN_STD_DEV to prevent division by zero."""
        sensor_data = {
            "sensor_a": {"current": 10.0, "baseline": 5.0, "std": 0.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        assert len(contributions) == 1
        assert contributions[0].weight > 0

    def test_negative_std_uses_absolute(self, explainer):
        """Negative std should use absolute value (or minimum)."""
        sensor_data = {
            "sensor_a": {"current": 10.0, "baseline": 5.0, "std": -1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        assert len(contributions) == 1
        assert contributions[0].weight > 0

    def test_zero_baseline_nonzero_current(self, explainer):
        """Zero baseline with nonzero current."""
        sensor_data = {
            "sensor_a": {"current": 5.0, "baseline": 0.0, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        assert len(contributions) == 1
        assert contributions[0].weight > 0


# ---------------------------------------------------------------------------
# TestWeightValidation
# ---------------------------------------------------------------------------
class TestWeightValidation:
    """Weight validation: must sum > 0."""

    def test_weights_always_sum_to_one(self, explainer):
        """Regardless of input, weights must sum to 1.0 (or empty)."""
        test_cases = [
            {"s1": {"current": 1.0, "baseline": 0.0, "std": 1.0, "is_signature": False}},
            {"s1": {"current": 0.0, "baseline": 0.0, "std": 0.0, "is_signature": False},
             "s2": {"current": 0.0, "baseline": 0.0, "std": 0.0, "is_signature": False}},
            {"s1": {"current": 100.0, "baseline": 50.0, "std": 0.001, "is_signature": True}},
        ]
        for sensor_data in test_cases:
            contributions = explainer.compute_contributions(sensor_data, [])
            if contributions:
                total = sum(c.weight for c in contributions)
                assert abs(total - 1.0) < 1e-6, f"Failed for {sensor_data}"

    def test_all_weights_non_negative(self, explainer):
        """All weights must be non-negative."""
        sensor_data = {
            "s1": {"current": -5.0, "baseline": 5.0, "std": 2.0, "is_signature": False},
            "s2": {"current": 10.0, "baseline": 3.0, "std": 1.0, "is_signature": True},
            "s3": {"current": 0.0, "baseline": 0.0, "std": 0.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, ["s2"])
        for c in contributions:
            assert c.weight >= 0.0, f"Negative weight for {c.sensor_name}: {c.weight}"


# ---------------------------------------------------------------------------
# TestExplanationResult
# ---------------------------------------------------------------------------
class TestExplanationResult:
    """Test the full explanation result."""

    def test_explain_returns_result(self, explainer, sample_sensor_data, signature_sensors):
        """explain() returns an ExplanationResult."""
        result = explainer.explain(
            signature_confidence=0.8,
            ml_score=0.7,
            sensor_data=sample_sensor_data,
            signature_sensors=signature_sensors,
        )
        assert result is not None
        assert hasattr(result, "reliability")
        assert hasattr(result, "sensor_contributions")

    def test_explain_reliability_correct(self, explainer, sample_sensor_data, signature_sensors):
        """Explain result has correct reliability."""
        result = explainer.explain(
            signature_confidence=0.8,
            ml_score=0.7,
            sensor_data=sample_sensor_data,
            signature_sensors=signature_sensors,
        )
        expected = 0.6 * 0.8 + 0.4 * 0.7
        assert abs(result.reliability - expected) < 1e-9

    def test_explain_contributions_sum_to_one(self, explainer, sample_sensor_data, signature_sensors):
        """Explain result contributions sum to 1.0."""
        result = explainer.explain(
            signature_confidence=0.8,
            ml_score=0.7,
            sensor_data=sample_sensor_data,
            signature_sensors=signature_sensors,
        )
        total = sum(c.weight for c in result.sensor_contributions)
        assert abs(total - 1.0) < 1e-6

    def test_explain_with_empty_sensors(self, explainer):
        """Explain with empty sensor data still works."""
        result = explainer.explain(
            signature_confidence=0.5,
            ml_score=0.5,
            sensor_data={},
            signature_sensors=[],
        )
        assert abs(result.reliability - 0.5) < 1e-9
        assert len(result.sensor_contributions) == 0


# ---------------------------------------------------------------------------
# TestReliabilityScoreDataclass
# ---------------------------------------------------------------------------
class TestReliabilityScoreDataclass:
    """Test ReliabilityScore dataclass."""

    def test_reliability_score_fields(self):
        """ReliabilityScore has required fields."""
        score = ReliabilityScore(
            reliability=0.76,
            signature_confidence=0.8,
            ml_score=0.7,
        )
        assert score.reliability == 0.76
        assert score.signature_confidence == 0.8
        assert score.ml_score == 0.7

    def test_reliability_score_bounded(self):
        """ReliabilityScore values should be in [0, 1]."""
        score = ReliabilityScore(
            reliability=0.5,
            signature_confidence=0.5,
            ml_score=0.5,
        )
        assert 0.0 <= score.reliability <= 1.0
        assert 0.0 <= score.signature_confidence <= 1.0
        assert 0.0 <= score.ml_score <= 1.0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Additional edge cases."""

    def test_very_small_std(self, explainer):
        """Very small std (but not zero) should work."""
        sensor_data = {
            "sensor_a": {"current": 5.1, "baseline": 5.0, "std": 0.0001, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        assert len(contributions) == 1
        assert contributions[0].weight > 0

    def test_very_large_deviation(self, explainer):
        """Very large deviation should not overflow."""
        sensor_data = {
            "sensor_a": {"current": 1e10, "baseline": 0.0, "std": 1.0, "is_signature": False},
            "sensor_b": {"current": 5.0, "baseline": 0.0, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, [])
        total = sum(c.weight for c in contributions)
        assert abs(total - 1.0) < 1e-6

    def test_multiple_signature_sensors(self, explainer):
        """Multiple signature sensors all get 1.5x boost."""
        sensor_data = {
            "sig_1": {"current": 6.0, "baseline": 5.0, "std": 1.0, "is_signature": True},
            "sig_2": {"current": 6.0, "baseline": 5.0, "std": 1.0, "is_signature": True},
            "normal": {"current": 6.0, "baseline": 5.0, "std": 1.0, "is_signature": False},
        }
        contributions = explainer.compute_contributions(sensor_data, ["sig_1", "sig_2"])
        total = sum(c.weight for c in contributions)
        assert abs(total - 1.0) < 1e-6

        sig1 = next(c for c in contributions if c.sensor_name == "sig_1")
        normal = next(c for c in contributions if c.sensor_name == "normal")
        # Each signature should have 1.5x the weight of normal
        assert abs(sig1.weight / normal.weight - 1.5) < 1e-6
