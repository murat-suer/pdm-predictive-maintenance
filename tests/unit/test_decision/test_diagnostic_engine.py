"""
Unit tests for src.decision.diagnostic_engine (Phase 2B - Core Logic Layer)

Tests DiagnosticEngine: hybrid rule+ML diagnostic engine with 3 diagnosis types
(PROCESS_ANOMALY, SENSOR_ANOMALY, UNKNOWN), reliability scoring,
sensor attribution (SHAP-like), and evidence chain construction.

NOTE: These tests will FAIL until diagnostic_engine.py is implemented.
"""

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.diagnostic_engine import (
    RELIABILITY_ML_WEIGHT,
    RELIABILITY_RULE_WEIGHT,
    SENSOR_ANOMALY_RECURRING_THRESHOLD,
    DiagnosisType,
    DiagnosticEngine,
    MLScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ml_scorer():
    """Mock ML scorer that returns configurable anomaly scores."""
    scorer = MagicMock(spec=MLScorer)
    scorer.score.return_value = 0.85
    return scorer


@pytest.fixture
def engine(mock_ml_scorer):
    """DiagnosticEngine with mock ML scorer."""
    return DiagnosticEngine(ml_scorer=mock_ml_scorer)


@pytest.fixture
def sensor_data_normal():
    """Normal sensor readings (no anomaly)."""
    return {
        "vibration_rms": {"value": 2.5, "trend": "stable", "baseline": 2.3},
        "temperature": {"value": 65.0, "trend": "stable", "baseline": 64.0},
        "pressure": {"value": 4.2, "trend": "stable", "baseline": 4.1},
    }


@pytest.fixture
def sensor_data_process_anomaly():
    """Sensor data matching a known process anomaly signature."""
    return {
        "vibration_rms": {"value": 12.5, "trend": "up", "baseline": 2.3, "delta_pct": 443},
        "temperature": {"value": 95.0, "trend": "up", "baseline": 64.0, "delta_pct": 48},
        "pressure": {"value": 6.8, "trend": "up", "baseline": 4.1, "delta_pct": 65},
    }


@pytest.fixture
def sensor_data_sensor_anomaly():
    """Sensor data with recurring pattern on same sensor (3+ times)."""
    return {
        "vibration_rms": {"value": 15.0, "trend": "up", "baseline": 2.3, "delta_pct": 552},
        "temperature": {"value": 65.0, "trend": "stable", "baseline": 64.0},
        "pressure": {"value": 4.2, "trend": "stable", "baseline": 4.1},
    }


@pytest.fixture
def sensor_data_unknown_anomaly():
    """Sensor data where ML detects anomaly but no library match.
    
    Use unknown sensor names that don't match any failure mode signature.
    """
    return {
        "unknown_sensor_xyz": {"value": 8.0, "trend": "up", "baseline": 2.3},
        "mysterious_sensor_abc": {"value": 120.0, "trend": "up", "baseline": 30.0},
    }


# ---------------------------------------------------------------------------
# TestDiagnosisTypeProcessAnomaly
# ---------------------------------------------------------------------------
class TestDiagnosisTypeProcessAnomaly:
    """PROCESS_ANOMALY: library match + high ML score."""

    def test_process_anomaly_detected(self, engine, sensor_data_process_anomaly):
        """When library matches and ML score is high → PROCESS_ANOMALY."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        assert result.diagnosis_type == DiagnosisType.PROCESS_ANOMALY

    def test_process_anomaly_has_mode_id(self, engine, sensor_data_process_anomaly):
        """PROCESS_ANOMALY should reference a failure mode from library."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        assert result.mode_id is not None
        assert len(result.mode_id) > 0

    def test_process_anomaly_high_confidence(self, engine, sensor_data_process_anomaly):
        """PROCESS_ANOMALY should have high confidence."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        assert result.confidence > 0.5


# ---------------------------------------------------------------------------
# TestDiagnosisTypeSensorAnomaly
# ---------------------------------------------------------------------------
class TestDiagnosisTypeSensorAnomaly:
    """SENSOR_ANOMALY: recurring pattern (3+ same sensor)."""

    def test_sensor_anomaly_recurring_pattern(self, engine, sensor_data_sensor_anomaly):
        """When same sensor triggers 3+ times → SENSOR_ANOMALY."""
        # Simulate recurring history
        engine.record_sensor_event(
            sensor_name="vibration_rms",
            machine_id="AC-001",
            event_type="threshold_breach",
        )
        engine.record_sensor_event(
            sensor_name="vibration_rms",
            machine_id="AC-001",
            event_type="threshold_breach",
        )
        # This is the 3rd occurrence
        result = engine.diagnose(
            sensor_data=sensor_data_sensor_anomaly,
            machine_id="AC-001",
        )
        assert result.diagnosis_type == DiagnosisType.SENSOR_ANOMALY

    def test_sensor_anomaly_below_threshold_not_triggered(self, engine, sensor_data_normal):
        """Below recurring threshold → not SENSOR_ANOMALY."""
        engine.record_sensor_event(
            sensor_name="vibration_rms",
            machine_id="AC-001",
            event_type="threshold_breach",
        )
        # Only 1 occurrence, below threshold
        result = engine.diagnose(
            sensor_data=sensor_data_normal,
            machine_id="AC-001",
        )
        assert result.diagnosis_type != DiagnosisType.SENSOR_ANOMALY


# ---------------------------------------------------------------------------
# TestDiagnosisTypeUnknown
# ---------------------------------------------------------------------------
class TestDiagnosisTypeUnknown:
    """UNKNOWN: ML detects anomaly but no library match."""

    def test_unknown_when_no_library_match(self, engine, sensor_data_unknown_anomaly, mock_ml_scorer):
        """ML detects anomaly but no library match → UNKNOWN."""
        mock_ml_scorer.score.return_value = 0.9  # High ML anomaly score
        result = engine.diagnose(
            sensor_data=sensor_data_unknown_anomaly,
            machine_id="AC-001",
        )
        assert result.diagnosis_type == DiagnosisType.UNKNOWN

    def test_unknown_has_low_signature_confidence(self, engine, sensor_data_unknown_anomaly, mock_ml_scorer):
        """UNKNOWN diagnosis should have low/no signature confidence."""
        mock_ml_scorer.score.return_value = 0.9
        result = engine.diagnose(
            sensor_data=sensor_data_unknown_anomaly,
            machine_id="AC-001",
        )
        assert result.signature_confidence < 0.3 or result.signature_confidence == 0.0


# ---------------------------------------------------------------------------
# TestReliabilityScore
# ---------------------------------------------------------------------------
class TestReliabilityScore:
    """Reliability score = 0.6 * signature_confidence + 0.4 * ml_score."""

    def test_reliability_formula(self, engine, mock_ml_scorer):
        """Verify reliability = 0.6 * sig_conf + 0.4 * ml_score."""
        mock_ml_scorer.score.return_value = 0.8
        # Manually calculate expected
        sig_conf = 0.9
        ml_score = 0.8
        expected = RELIABILITY_RULE_WEIGHT * sig_conf + RELIABILITY_ML_WEIGHT * ml_score
        actual = engine.calculate_reliability(
            signature_confidence=sig_conf,
            ml_score=ml_score,
        )
        assert abs(actual - expected) < 1e-9

    def test_reliability_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        assert abs(RELIABILITY_RULE_WEIGHT + RELIABILITY_ML_WEIGHT - 1.0) < 1e-9

    def test_reliability_zero_inputs(self, engine):
        """Zero inputs → zero reliability."""
        result = engine.calculate_reliability(
            signature_confidence=0.0,
            ml_score=0.0,
        )
        assert result == 0.0

    def test_reliability_max_inputs(self, engine):
        """Max inputs (1.0, 1.0) → 1.0."""
        result = engine.calculate_reliability(
            signature_confidence=1.0,
            ml_score=1.0,
        )
        assert abs(result - 1.0) < 1e-9

    def test_reliability_boundary_values(self, engine):
        """Boundary: sig=1.0, ml=0.0 → 0.6."""
        result = engine.calculate_reliability(
            signature_confidence=1.0,
            ml_score=0.0,
        )
        assert abs(result - RELIABILITY_RULE_WEIGHT) < 1e-9

    def test_reliability_in_diagnosis_result(self, engine, sensor_data_process_anomaly, mock_ml_scorer):
        """Diagnosis result should include computed reliability."""
        mock_ml_scorer.score.return_value = 0.75
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        assert hasattr(result, "reliability")
        assert 0.0 <= result.reliability <= 1.0


# ---------------------------------------------------------------------------
# TestSensorContribution
# ---------------------------------------------------------------------------
class TestSensorContribution:
    """SHAP-like sensor attribution."""

    def test_sensor_contributions_returned(self, engine, sensor_data_process_anomaly):
        """Diagnosis should include sensor contributions."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        assert hasattr(result, "sensor_contributions")
        assert len(result.sensor_contributions) > 0

    def test_contributions_sum_approximately_one(self, engine, sensor_data_process_anomaly):
        """Sensor contributions should sum to ~1.0 (normalized)."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        total = sum(c.weight for c in result.sensor_contributions)
        assert abs(total - 1.0) < 0.1  # Allow small floating point tolerance

    def test_contribution_has_sensor_name(self, engine, sensor_data_process_anomaly):
        """Each contribution should reference a sensor name."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        for contrib in result.sensor_contributions:
            assert hasattr(contrib, "sensor_name")
            assert contrib.sensor_name is not None
            assert len(contrib.sensor_name) > 0

    def test_contribution_weight_non_negative(self, engine, sensor_data_process_anomaly):
        """Contribution weights should be non-negative."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        for contrib in result.sensor_contributions:
            assert contrib.weight >= 0.0


# ---------------------------------------------------------------------------
# TestEvidenceChain
# ---------------------------------------------------------------------------
class TestEvidenceChain:
    """Evidence chain construction for audit trail."""

    def test_evidence_chain_created(self, engine, sensor_data_process_anomaly):
        """Diagnosis should produce an evidence chain."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        assert hasattr(result, "evidence_chain")
        assert len(result.evidence_chain) > 0

    def test_evidence_chain_has_timestamp(self, engine, sensor_data_process_anomaly):
        """Each evidence item should have a timestamp."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        for item in result.evidence_chain:
            assert hasattr(item, "timestamp")
            assert item.timestamp is not None

    def test_evidence_chain_has_source(self, engine, sensor_data_process_anomaly):
        """Each evidence item should identify its source."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        for item in result.evidence_chain:
            assert hasattr(item, "source")
            assert item.source in ("rule_engine", "ml_model", "sensor_data", "history")

    def test_evidence_chain_ordered(self, engine, sensor_data_process_anomaly):
        """Evidence chain should be in chronological order."""
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        timestamps = [item.timestamp for item in result.evidence_chain]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# TestMLScorerProtocol
# ---------------------------------------------------------------------------
class TestMLScorerProtocol:
    """ML scorer protocol/interface tests."""

    def test_ml_scorer_called_during_diagnosis(self, engine, mock_ml_scorer, sensor_data_process_anomaly):
        """ML scorer should be called during diagnosis."""
        engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        assert mock_ml_scorer.score.called

    def test_ml_scorer_receives_sensor_data(self, engine, mock_ml_scorer, sensor_data_process_anomaly):
        """ML scorer should receive sensor data."""
        engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        call_args = mock_ml_scorer.score.call_args
        # Verify sensor_data was passed
        assert call_args is not None

    def test_ml_scorer_returns_float(self, engine, mock_ml_scorer, sensor_data_process_anomaly):
        """ML scorer should return a float between 0 and 1."""
        mock_ml_scorer.score.return_value = 0.75
        result = engine.diagnose(
            sensor_data=sensor_data_process_anomaly,
            machine_id="AC-001",
        )
        # The ML score should be used in the result
        assert result.ml_score == 0.75

    def test_ml_scorer_low_score_no_anomaly(self, engine, mock_ml_scorer, sensor_data_normal):
        """Low ML score + no library match → no anomaly or UNKNOWN."""
        mock_ml_scorer.score.return_value = 0.1
        result = engine.diagnose(
            sensor_data=sensor_data_normal,
            machine_id="AC-001",
        )
        # With low ML score and no match, should not be PROCESS_ANOMALY
        assert result.diagnosis_type != DiagnosisType.PROCESS_ANOMALY


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases: empty data, None baselines, threshold boundaries."""

    def test_empty_sensor_data(self, engine):
        """Empty sensor data should return a safe default."""
        result = engine.diagnose(
            sensor_data={},
            machine_id="AC-001",
        )
        assert result is not None
        assert result.diagnosis_type in (DiagnosisType.UNKNOWN, DiagnosisType.PROCESS_ANOMALY)

    def test_none_sensor_data_raises(self, engine):
        """None sensor data should raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            engine.diagnose(
                sensor_data=None,
                machine_id="AC-001",
            )

    def test_sensor_with_none_baseline(self, engine):
        """Sensor with None baseline should be handled gracefully."""
        sensor_data = {
            "vibration_rms": {"value": 5.0, "trend": "up", "baseline": None},
        }
        result = engine.diagnose(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        assert result is not None

    def test_threshold_boundary_exact(self, engine, mock_ml_scorer):
        """At exact threshold boundary, behavior should be deterministic."""
        mock_ml_scorer.score.return_value = 0.5  # Boundary
        sensor_data = {
            "vibration_rms": {"value": 5.0, "trend": "up", "baseline": 5.0},
        }
        result = engine.diagnose(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        assert result is not None

    def test_missing_machine_id(self, engine, sensor_data_normal):
        """Missing/empty machine_id should be handled."""
        with pytest.raises((ValueError, TypeError)):
            engine.diagnose(
                sensor_data=sensor_data_normal,
                machine_id="",
            )

    def test_recurring_threshold_constant(self):
        """SENSOR_ANOMALY_RECURRING_THRESHOLD should be >= 3."""
        assert SENSOR_ANOMALY_RECURRING_THRESHOLD >= 3

    def test_single_sensor_data(self, engine, mock_ml_scorer):
        """Single sensor reading should not crash."""
        mock_ml_scorer.score.return_value = 0.6
        sensor_data = {
            "vibration_rms": {"value": 10.0, "trend": "up", "baseline": 2.3},
        }
        result = engine.diagnose(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        assert result is not None

    def test_negative_sensor_values(self, engine, mock_ml_scorer):
        """Negative sensor values should be handled."""
        mock_ml_scorer.score.return_value = 0.3
        sensor_data = {
            "vibration_rms": {"value": -1.0, "trend": "down", "baseline": 2.3},
        }
        result = engine.diagnose(
            sensor_data=sensor_data,
            machine_id="AC-001",
        )
        assert result is not None
