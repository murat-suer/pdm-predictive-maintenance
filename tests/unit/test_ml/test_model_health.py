"""Unit tests for src.ml.model_health module.

Tests CanaryProbeSystem and DriftDetector with mocked dependencies.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.ml.model_health import CanaryProbeSystem, DriftDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_probe_history():
    """Clear class-level probe history before each test."""
    CanaryProbeSystem._probe_history = []
    yield
    CanaryProbeSystem._probe_history = []


@pytest.fixture
def mock_anomaly_detector():
    """Create a mock AnomalyDetector with feature_names and predict."""
    detector = MagicMock()
    detector.feature_names = [
        "vibration_rms_value",
        "vibration_rms_rolling_mean_5m",
        "vibration_rms_rolling_std_5m",
        "vibration_rms_rate_of_change",
        "vibration_rms_z_score",
        "vibration_rms_shift_adj_z",
        "bearing_temp_value",
        "bearing_temp_rolling_mean_5m",
        "bearing_temp_rolling_std_5m",
        "bearing_temp_rate_of_change",
        "bearing_temp_z_score",
        "bearing_temp_shift_adj_z",
    ]
    detector.predict.return_value = {"is_anomaly": True, "anomaly_score": 0.85}
    return detector


@pytest.fixture
def mock_machine_configs():
    """Return a minimal MACHINE_CONFIGS dict for testing."""
    return {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    }


@pytest.fixture
def mock_db_session():
    """Create a mock SQLAlchemy session."""
    session = MagicMock()
    return session


# ---------------------------------------------------------------------------
# CanaryProbeSystem Tests
# ---------------------------------------------------------------------------


class TestCanaryProbeSystemInit:
    """Test CanaryProbeSystem initialization and class attributes."""

    def test_probe_scenarios_defined(self):
        """All machine types have probe scenarios."""
        assert "AC" in CanaryProbeSystem.PROBE_SCENARIOS
        assert "HX" in CanaryProbeSystem.PROBE_SCENARIOS
        assert "CM" in CanaryProbeSystem.PROBE_SCENARIOS

    def test_ac_scenarios_have_names(self):
        """AC scenarios each have a name."""
        for scenario in CanaryProbeSystem.PROBE_SCENARIOS["AC"]:
            assert "name" in scenario

    def test_hx_scenarios_have_names(self):
        """HX scenarios each have a name."""
        for scenario in CanaryProbeSystem.PROBE_SCENARIOS["HX"]:
            assert "name" in scenario

    def test_cm_scenarios_have_names(self):
        """CM scenarios each have a name."""
        for scenario in CanaryProbeSystem.PROBE_SCENARIOS["CM"]:
            assert "name" in scenario

    def test_probe_history_starts_empty(self):
        """Probe history is empty after fixture clears it."""
        assert CanaryProbeSystem._probe_history == []


class TestCanaryProbeSystemRunProbe:
    """Test canary probe execution."""

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_run_probe_detects_anomaly(self, mock_anomaly_detector):
        """Probe succeeds when model detects synthetic anomaly."""
        mock_anomaly_detector.predict.return_value = {"is_anomaly": True, "anomaly_score": 0.9}
        probe = CanaryProbeSystem()
        result = probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        assert result["detected"] is True
        assert result["success"] is True
        assert result["expected"] is True
        assert result["recalibration_triggered"] is False
        assert result["machine_id"] == "AC-201"
        assert result["probe_type"] == "AC"
        assert result["triggered_by"] == "SCHEDULED"

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_run_probe_failure_triggers_recalibration(self, mock_anomaly_detector):
        """Probe failure triggers recalibration flag."""
        mock_anomaly_detector.predict.return_value = {"is_anomaly": False, "anomaly_score": 0.1}
        probe = CanaryProbeSystem()
        result = probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        assert result["detected"] is False
        assert result["success"] is False
        assert result["recalibration_triggered"] is True

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_run_probe_adds_to_history(self, mock_anomaly_detector):
        """Probe result is added to _probe_history."""
        probe = CanaryProbeSystem()
        probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        assert len(CanaryProbeSystem._probe_history) == 1
        assert CanaryProbeSystem._probe_history[0]["machine_id"] == "AC-201"

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_run_probe_history_bounded_at_500(self, mock_anomaly_detector):
        """Probe history is capped at 500 entries (instance attribute after truncation)."""
        probe = CanaryProbeSystem()
        # Pre-fill class-level history to 500
        CanaryProbeSystem._probe_history = [{"probe_id": str(i)} for i in range(500)]

        probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        # After append + truncation, instance attribute has 500 entries
        assert len(probe._probe_history) == 500

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_run_probe_with_drift_alert_trigger(self, mock_anomaly_detector):
        """Probe can be triggered by DRIFT_ALERT."""
        probe = CanaryProbeSystem()
        result = probe.run_probe("AC-201", "AC", mock_anomaly_detector, triggered_by="DRIFT_ALERT")

        assert result["triggered_by"] == "DRIFT_ALERT"

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "UNKNOWN-001": {"sensors": {}}
    })
    def test_run_probe_unknown_machine_type(self, mock_anomaly_detector):
        """Probe with unknown machine type uses empty scenario."""
        mock_anomaly_detector.predict.return_value = {"is_anomaly": False, "anomaly_score": 0.0}
        probe = CanaryProbeSystem()
        result = probe.run_probe("UNKNOWN-001", "UNKNOWN", mock_anomaly_detector)

        assert result["scenario"] is None
        assert result["detected"] is False

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_run_probe_result_has_timestamps(self, mock_anomaly_detector):
        """Probe result includes started_at and completed_at timestamps."""
        probe = CanaryProbeSystem()
        result = probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        assert isinstance(result["started_at"], datetime)
        assert isinstance(result["completed_at"], datetime)
        assert result["completed_at"] >= result["started_at"]

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_run_probe_result_has_probe_id(self, mock_anomaly_detector):
        """Probe result includes a UUID probe_id."""
        probe = CanaryProbeSystem()
        result = probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        assert "probe_id" in result
        assert len(result["probe_id"]) == 36  # UUID format

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_run_probe_notes_include_score(self, mock_anomaly_detector):
        """Probe notes include anomaly score when result is available."""
        mock_anomaly_detector.predict.return_value = {"is_anomaly": True, "anomaly_score": 0.876}
        probe = CanaryProbeSystem()
        result = probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        assert "0.876" in result["notes"]


class TestCanaryProbeSystemBuildSyntheticFeatures:
    """Test _build_synthetic_features method."""

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_affected_sensor_gets_multiplied_value(self, mock_anomaly_detector):
        """Affected sensor with _mult key gets multiplied value."""
        probe = CanaryProbeSystem()
        scenario = {"name": "bearing_fault", "vibration_rms_mult": 2.5}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        # vibration_rms is affected: value = 2.5 * 2.5 = 6.25
        assert features["vibration_rms_value"] == pytest.approx(6.25)

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_affected_sensor_gets_delta_value(self, mock_anomaly_detector):
        """Affected sensor with _delta key gets added delta."""
        probe = CanaryProbeSystem()
        scenario = {"name": "overheat", "bearing_temp_delta": 15}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        # bearing_temp is affected: value = 65.0 + 15 = 80.0
        assert features["bearing_temp_value"] == pytest.approx(80.0)

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
                "bearing_temp": {"nominal_mu": 65.0, "nominal_sigma": 3.0},
            }
        }
    })
    def test_unaffected_sensor_keeps_nominal(self, mock_anomaly_detector):
        """Unaffected sensor keeps nominal value."""
        probe = CanaryProbeSystem()
        scenario = {"name": "vibration_issue", "vibration_rms_mult": 3.0}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        # bearing_temp is NOT affected: value = 65.0 (nominal)
        assert features["bearing_temp_value"] == pytest.approx(65.0)

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_affected_sensor_rolling_mean_shifted(self, mock_anomaly_detector):
        """Affected sensor rolling_mean_5m is shifted."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test", "vibration_rms_mult": 2.0}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        # mu + 0.5 * sigma = 2.5 + 0.5 * 0.15 = 2.575
        assert features["vibration_rms_rolling_mean_5m"] == pytest.approx(2.575)

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_affected_sensor_rolling_std_doubled(self, mock_anomaly_detector):
        """Affected sensor rolling_std_5m is doubled."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test", "vibration_rms_mult": 2.0}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        # 2.0 * sigma = 2.0 * 0.15 = 0.30
        assert features["vibration_rms_rolling_std_5m"] == pytest.approx(0.30)

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_affected_sensor_z_score_high(self, mock_anomaly_detector):
        """Affected sensor z_score is set to 4.0."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test", "vibration_rms_mult": 2.0}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        assert features["vibration_rms_z_score"] == 4.0

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_affected_sensor_shift_adj_z_higher(self, mock_anomaly_detector):
        """Affected sensor shift_adj_z is set to 5.0 (higher than z_score)."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test", "vibration_rms_mult": 2.0}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        assert features["vibration_rms_shift_adj_z"] == 5.0

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_unaffected_sensor_z_score_zero(self, mock_anomaly_detector):
        """Unaffected sensor z_score is 0.0."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test"}  # no vibration_rms keys
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        assert features["vibration_rms_z_score"] == 0.0
        assert features["vibration_rms_shift_adj_z"] == 0.0

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_unaffected_sensor_rate_of_change_zero(self, mock_anomaly_detector):
        """Unaffected sensor rate_of_change is 0.0."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test"}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        assert features["vibration_rms_rate_of_change"] == 0.0

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_affected_sensor_rate_of_change_nonzero(self, mock_anomaly_detector):
        """Affected sensor rate_of_change = value - mu."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test", "vibration_rms_mult": 3.0}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        # value = 2.5 * 3.0 = 7.5; rate_of_change = 7.5 - 2.5 = 5.0
        assert features["vibration_rms_rate_of_change"] == pytest.approx(5.0)

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_empty_feature_names_returns_empty(self, mock_anomaly_detector):
        """Empty feature_names returns empty features dict."""
        mock_anomaly_detector.feature_names = []
        probe = CanaryProbeSystem()
        scenario = {"name": "test", "vibration_rms_mult": 2.0}
        features = probe._build_synthetic_features("AC-201", "AC", scenario, mock_anomaly_detector)

        assert features == {}

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {})
    def test_unknown_machine_id_returns_empty(self, mock_anomaly_detector):
        """Unknown machine_id returns empty features dict."""
        probe = CanaryProbeSystem()
        scenario = {"name": "test"}
        features = probe._build_synthetic_features("UNKNOWN", "AC", scenario, mock_anomaly_detector)

        assert features == {}


class TestCanaryProbeSystemGetModelMetrics:
    """Test get_model_metrics class method."""

    def test_metrics_without_db_session(self):
        """Metrics work without a DB session (no anomaly rate from DB)."""
        metrics = CanaryProbeSystem.get_model_metrics(db_session=None)

        assert metrics["anomaly_rate"] == 0.0
        assert metrics["anomaly_count"] == 0
        assert metrics["total_readings"] == 0
        assert metrics["canary_detection_rate"] is None
        assert metrics["total_canary_probes"] == 0

    @patch("src.data_generator.machines.MACHINE_CONFIGS", {
        "AC-201": {
            "sensors": {
                "vibration_rms": {"nominal_mu": 2.5, "nominal_sigma": 0.15},
            }
        }
    })
    def test_metrics_with_probe_history(self, mock_anomaly_detector):
        """Metrics reflect probe history correctly."""
        probe = CanaryProbeSystem()
        # Run 2 probes: 1 success, 1 failure
        mock_anomaly_detector.predict.side_effect = [
            {"is_anomaly": True, "anomaly_score": 0.9},
            {"is_anomaly": False, "anomaly_score": 0.1},
        ]
        probe.run_probe("AC-201", "AC", mock_anomaly_detector)
        probe.run_probe("AC-201", "AC", mock_anomaly_detector)

        metrics = CanaryProbeSystem.get_model_metrics(db_session=None)

        assert metrics["total_canary_probes"] == 2
        assert metrics["canary_detected"] == 1
        assert metrics["canary_missed"] == 1
        assert metrics["canary_detection_rate"] == 0.5
        assert metrics["last_canary_result"]["detected"] is False

    def test_metrics_last_probe_is_none_when_empty(self):
        """last_canary_result is None when no probes have run."""
        metrics = CanaryProbeSystem.get_model_metrics(db_session=None)
        assert metrics["last_canary_result"] is None


# ---------------------------------------------------------------------------
# DriftDetector Tests
# ---------------------------------------------------------------------------


class TestDriftDetectorInit:
    """Test DriftDetector class attributes."""

    def test_thresholds_defined(self):
        """All drift thresholds are defined."""
        assert DriftDetector.ANOMALY_RATE_WINDOW_HOURS == 24
        assert DriftDetector.ANOMALY_RATE_THRESHOLD == 0.15
        assert DriftDetector.ALIGNMENT_THRESHOLD == 0.30
        assert DriftDetector.FP_RATE_THRESHOLD == 0.20


class TestDriftDetectorAnomalyRateDrift:
    """Test check_anomaly_rate_drift method."""

    def test_no_drift_when_rate_below_threshold(self, mock_db_session):
        """No drift when anomaly rate is below 15%."""
        # Mock: 100 total readings, 10 anomalies = 10% rate
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        # First call: total count = 100
        # Second call: anomaly count = 10
        query_chain.scalar.side_effect = [100, 10]

        detector = DriftDetector()
        result = detector.check_anomaly_rate_drift("AC-201", mock_db_session)

        assert result["drift_detected"] is False
        assert result["anomaly_rate"] == 0.1
        assert result["anomaly_count"] == 10
        assert result["total_readings"] == 100
        assert "DRIFT ALERT" not in result["message"]

    def test_drift_when_rate_above_threshold(self, mock_db_session):
        """Drift detected when anomaly rate exceeds 15%."""
        # Mock: 100 total, 20 anomalies = 20% rate
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.scalar.side_effect = [100, 20]

        detector = DriftDetector()
        result = detector.check_anomaly_rate_drift("AC-201", mock_db_session)

        assert result["drift_detected"] is True
        assert result["anomaly_rate"] == 0.2
        assert "DRIFT ALERT" in result["message"]

    def test_handles_zero_total_readings(self, mock_db_session):
        """Handles zero total readings gracefully (defaults to 1)."""
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        # scalar returns 0 (falsy) -> or 1 -> total = 1
        query_chain.scalar.side_effect = [0, 0]

        detector = DriftDetector()
        result = detector.check_anomaly_rate_drift("AC-201", mock_db_session)

        assert result["total_readings"] == 1
        assert result["anomaly_count"] == 0
        assert result["anomaly_rate"] == 0.0
        assert result["drift_detected"] is False


class TestDriftDetectorOperatorAlignment:
    """Test check_operator_alignment method."""

    def test_no_decisions_returns_no_drift(self, mock_db_session):
        """No decisions in window returns 100% alignment, no drift."""
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = []

        detector = DriftDetector()
        result = detector.check_operator_alignment("AC-201", mock_db_session)

        assert result["drift_detected"] is False
        assert result["operator_alignment_pct"] == 100.0
        assert result["override_count"] == 0
        assert result["total_decisions"] == 0

    def test_low_override_rate_no_drift(self, mock_db_session):
        """Low override rate (<30%) does not trigger drift."""
        # 10 decisions, 2 overridden = 20% override = 80% alignment
        decisions = []
        for i in range(10):
            d = MagicMock()
            d.overridden = (i < 2)
            decisions.append(d)

        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = decisions

        detector = DriftDetector()
        result = detector.check_operator_alignment("AC-201", mock_db_session)

        assert result["drift_detected"] is False
        assert result["operator_alignment_pct"] == 80.0
        assert result["override_count"] == 2
        assert result["total_decisions"] == 10

    def test_high_override_rate_triggers_drift(self, mock_db_session):
        """High override rate (>30%) triggers drift."""
        # 10 decisions, 5 overridden = 50% override = 50% alignment
        decisions = []
        for i in range(10):
            d = MagicMock()
            d.overridden = (i < 5)
            decisions.append(d)

        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = decisions

        detector = DriftDetector()
        result = detector.check_operator_alignment("AC-201", mock_db_session)

        assert result["drift_detected"] is True
        assert result["operator_alignment_pct"] == 50.0
        assert "DRIFT ALERT" in result["message"]


class TestDriftDetectorMaintenanceValidation:
    """Test check_maintenance_validation method."""

    def test_no_maintenance_returns_no_drift(self, mock_db_session):
        """No maintenance logs returns 0% FP rate, no drift."""
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = []

        detector = DriftDetector()
        result = detector.check_maintenance_validation("AC-201", mock_db_session)

        assert result["drift_detected"] is False
        assert result["false_positive_rate"] == 0.0
        assert result["fp_count"] == 0
        assert result["total_maintenance"] == 0

    def test_low_fp_rate_no_drift(self, mock_db_session):
        """Low FP rate (<20%) does not trigger drift."""
        # 10 logs, 1 fault_found=False = 10% FP rate
        logs = []
        for i in range(10):
            log = MagicMock()
            log.fault_found = (i != 0)  # first one is FP
            logs.append(log)

        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = logs

        detector = DriftDetector()
        result = detector.check_maintenance_validation("AC-201", mock_db_session)

        assert result["drift_detected"] is False
        assert result["false_positive_rate"] == 0.1
        assert result["fp_count"] == 1
        assert result["total_maintenance"] == 10

    def test_high_fp_rate_triggers_drift(self, mock_db_session):
        """High FP rate (>20%) triggers drift."""
        # 10 logs, 5 fault_found=False = 50% FP rate
        logs = []
        for i in range(10):
            log = MagicMock()
            log.fault_found = (i >= 5)  # first 5 are FP
            logs.append(log)

        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.all.return_value = logs

        detector = DriftDetector()
        result = detector.check_maintenance_validation("AC-201", mock_db_session)

        assert result["drift_detected"] is True
        assert result["false_positive_rate"] == 0.5
        assert result["fp_count"] == 5
        assert "DRIFT ALERT" in result["message"]


class TestDriftDetectorRunAllChecks:
    """Test run_all_checks method."""

    def test_no_drift_when_all_checks_pass(self, mock_db_session):
        """No drift when all 3 checks pass."""
        # Mock anomaly rate: 100 total, 5 anomalies = 5% (below 15%)
        # Mock alignment: no decisions
        # Mock maintenance: no logs
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.scalar.side_effect = [100, 5]  # total, anomaly_count
        query_chain.all.side_effect = [[], []]  # decisions, maintenance logs

        detector = DriftDetector()
        result = detector.run_all_checks("AC-201", mock_db_session)

        assert result["drift_detected"] is False
        assert result["machine_id"] == "AC-201"
        assert "checked_at" in result
        assert "anomaly_rate" in result
        assert "alignment" in result
        assert "fp_validation" in result

    def test_drift_when_anomaly_rate_high(self, mock_db_session):
        """Drift detected when anomaly rate check fails."""
        # Mock anomaly rate: 100 total, 20 anomalies = 20% (above 15%)
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.scalar.side_effect = [100, 20]
        query_chain.all.side_effect = [[], []]

        detector = DriftDetector()
        result = detector.run_all_checks("AC-201", mock_db_session)

        assert result["drift_detected"] is True

    def test_drift_when_alignment_low(self, mock_db_session):
        """Drift detected when operator alignment check fails."""
        # Mock anomaly rate: 100 total, 5 anomalies = 5% (OK)
        # Mock alignment: 10 decisions, 5 overridden = 50% override (FAIL)
        # Mock maintenance: no logs
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.scalar.side_effect = [100, 5]

        decisions = []
        for i in range(10):
            d = MagicMock()
            d.overridden = (i < 5)
            decisions.append(d)
        query_chain.all.side_effect = [decisions, []]

        detector = DriftDetector()
        result = detector.run_all_checks("AC-201", mock_db_session)

        assert result["drift_detected"] is True

    def test_drift_when_fp_rate_high(self, mock_db_session):
        """Drift detected when maintenance validation check fails."""
        # Mock anomaly rate: 100 total, 5 anomalies = 5% (OK)
        # Mock alignment: no decisions (OK)
        # Mock maintenance: 10 logs, 5 FP = 50% (FAIL)
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.scalar.side_effect = [100, 5]

        logs = []
        for i in range(10):
            log = MagicMock()
            log.fault_found = (i >= 5)
            logs.append(log)
        query_chain.all.side_effect = [[], logs]

        detector = DriftDetector()
        result = detector.run_all_checks("AC-201", mock_db_session)

        assert result["drift_detected"] is True

    def test_result_has_iso_timestamp(self, mock_db_session):
        """run_all_checks result includes ISO format timestamp."""
        query_chain = MagicMock()
        mock_db_session.query.return_value = query_chain
        query_chain.filter.return_value = query_chain
        query_chain.scalar.side_effect = [100, 5]
        query_chain.all.side_effect = [[], []]

        detector = DriftDetector()
        result = detector.run_all_checks("AC-201", mock_db_session)

        # Should be parseable as ISO datetime
        parsed = datetime.fromisoformat(result["checked_at"])
        assert isinstance(parsed, datetime)
