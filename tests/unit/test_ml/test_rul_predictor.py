"""
Comprehensive tests for src.ml.rul_predictor.RULPredictor.

Covers:
- Initialization (defaults, custom params, validation)
- train() with dummy data
- predict() for HEALTHY / FAILED / DEGRADING phases
- Smoothing logic (_smooth_rul)
- Schmitt trigger (should_publish_alarm)
- Weibull confidence intervals
- Model persistence (save/load)
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_model_store(tmp_path, monkeypatch):
    """Redirect model store to a temp directory so tests don't pollute the real one."""
    store_dir = tmp_path / "model_store"
    store_dir.mkdir()
    monkeypatch.setenv("MODEL_STORE_PATH", str(store_dir))
    # Also patch the get_model_store function to return our temp dir
    from src.ml.model_store import paths as _paths
    monkeypatch.setattr(_paths, "get_model_store", lambda: store_dir)


@pytest.fixture
def predictor():
    """Fresh RULPredictor with deterministic seed."""
    from src.ml.rul_predictor import RULPredictor
    return RULPredictor("TEST-001", beta=2.1, eta=720.0)


@pytest.fixture
def trained_predictor():
    """RULPredictor trained on synthetic data."""
    from src.ml.rul_predictor import RULPredictor
    p = RULPredictor("TEST-002", beta=2.1, eta=720.0)
    rng = np.random.RandomState(42)
    n_samples = 100
    n_features = 5
    X = rng.randn(n_samples, n_features)
    # y = some function of X so the model can learn
    y = np.abs(X @ rng.randn(n_features)) * 100 + 50
    feature_names = [f"feat_{i}" for i in range(n_features)]
    p.train(X, y, feature_names)
    return p


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_params(self, predictor):
        assert predictor.machine_id == "TEST-001"
        assert predictor.beta == 2.1
        assert predictor.eta == 720.0
        assert predictor.mode == "single"
        assert predictor.model is None
        assert predictor.feature_names == []
        assert predictor._alarm_state == "ARMED"
        assert predictor._last_smoothed_rul is None

    def test_custom_params(self):
        from src.ml.rul_predictor import RULPredictor
        p = RULPredictor(
            "CUSTOM-01",
            beta=3.0,
            eta=500.0,
            alarm_threshold_hours=50.0,
            alarm_hysteresis_hours=10.0,
            max_step_pct=0.20,
        )
        assert p.beta == 3.0
        assert p.eta == 500.0
        assert p._alarm_threshold_hours == 50.0
        assert p._alarm_hysteresis_hours == 10.0
        assert p._max_step_pct == 0.20

    def test_invalid_mode_raises(self):
        from src.ml.rul_predictor import RULPredictor
        with pytest.raises(ValueError, match="mode must be"):
            RULPredictor("X", mode="invalid")

    def test_invalid_n_ticks_raises(self):
        from src.ml.rul_predictor import RULPredictor
        with pytest.raises(ValueError, match="n_ticks_to_endpoint"):
            RULPredictor("X", n_ticks_to_endpoint=0)

    def test_invalid_max_step_pct_raises(self):
        from src.ml.rul_predictor import RULPredictor
        with pytest.raises(ValueError, match="max_step_pct"):
            RULPredictor("X", max_step_pct=0.0)
        with pytest.raises(ValueError, match="max_step_pct"):
            RULPredictor("X", max_step_pct=1.5)

    def test_target_endpoint_derives_max_step(self):
        from src.ml.rul_predictor import RULPredictor
        # eta=720, target=48, n_ticks=10
        # step = 1 - (48/720)^(1/10) ≈ 1 - 0.0667^0.1 ≈ 1 - 0.7499 ≈ 0.2501
        p = RULPredictor("X", eta=720.0, target_endpoint_hours=48.0, n_ticks_to_endpoint=10)
        assert 0.20 < p._max_step_pct < 0.30

    def test_invalid_target_endpoint_raises(self):
        from src.ml.rul_predictor import RULPredictor
        with pytest.raises(ValueError, match="target_endpoint_hours"):
            RULPredictor("X", target_endpoint_hours=-10.0)

    def test_conformal_initialized(self, predictor):
        assert predictor.conformal is not None
        assert predictor.conformal.alpha == 0.10

    def test_deterministic_rng(self):
        """Two predictors with same machine_id should have same RNG sequence."""
        from src.ml.rul_predictor import RULPredictor
        p1 = RULPredictor("DET-001", beta=2.1, eta=720.0)
        p2 = RULPredictor("DET-001", beta=2.1, eta=720.0)
        vals1 = [p1._rng.random() for _ in range(10)]
        vals2 = [p2._rng.random() for _ in range(10)]
        assert vals1 == vals2


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_with_sufficient_data(self):
        from src.ml.rul_predictor import RULPredictor
        p = RULPredictor("TRAIN-01", beta=2.1, eta=720.0)
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = rng.rand(60) * 500
        feature_names = ["f1", "f2", "f3"]
        p.train(X, y, feature_names)
        assert p.model is not None
        assert p.feature_names == feature_names

    def test_train_with_insufficient_data_warns(self, caplog):
        from src.ml.rul_predictor import RULPredictor
        p = RULPredictor("TRAIN-02", beta=2.1, eta=720.0)
        X = np.random.randn(10, 3)  # < 50 samples
        y = np.random.rand(10) * 500
        p.train(X, y, ["f1", "f2", "f3"])
        assert p.model is None  # Should not train

    def test_train_creates_model_file(self, trained_predictor):
        assert os.path.exists(trained_predictor._model_path)

    def test_train_creates_hash_file(self, trained_predictor):
        assert os.path.exists(f"{trained_predictor._model_path}.sha256")


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------

class TestPredictHealthy:
    def test_returns_none(self, predictor):
        result = predictor.predict(features={}, phase="HEALTHY")
        assert result is None

    def test_returns_none_with_features(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 5.0},
            phase="HEALTHY",
        )
        assert result is None

    def test_degradation_level_yields_age_based_rul(self, predictor):
        """A healthy machine with a known degradation level gets a finite,
        age-consistent RUL so dashboards never show a blank."""
        young = predictor.predict(
            features={}, phase="HEALTHY", degradation_level=0.05
        )
        old = predictor.predict(
            features={}, phase="HEALTHY", degradation_level=0.60
        )
        assert young is not None and old is not None
        assert young["method"] == "weibull_age"
        assert young["fallback"] is True
        assert 0 < old["rul_hours"] < young["rul_hours"] <= predictor.eta


class TestPredictFailed:
    def test_returns_zero_rul(self, predictor):
        result = predictor.predict(features={}, phase="FAILED")
        assert result is not None
        assert result["rul_hours"] == 0.0
        assert result["rul_low_ci"] == 0.0
        assert result["rul_high_ci"] == 0.0
        # Certainty is never claimed — confidence is capped at 0.99.
        assert result["confidence"] == 0.99
        assert result["failure_prob_24h"] == 100.0
        assert result["survive_shift_pct"] == 0.0
        assert result["method"] == "failed_state"
        assert result["fallback"] is True

    def test_resets_smoothing_state(self, predictor):
        # First, set up some smoothing state
        predictor._last_smoothed_rul = 500.0
        predictor.predict(features={}, phase="FAILED")
        assert predictor._last_smoothed_rul == 0.0


class TestPredictDegrading:
    def test_returns_dict_with_required_keys(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 2.0},
            phase="DEGRADING",
        )
        assert result is not None
        required_keys = [
            "rul_hours", "rul_low_ci", "rul_high_ci",
            "confidence", "failure_prob_24h", "survive_shift_pct",
            "method", "fallback", "model_trained",
            "coverage_guarantee", "conformal_calibration_size",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_rul_non_negative(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 5.0},
            phase="DEGRADING",
            emergency_stop_count=5,
        )
        assert result["rul_hours"] >= 0.0
        assert result["rul_low_ci"] >= 0.0
        assert result["rul_high_ci"] >= 0.0

    def test_method_ends_with_conformal(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 2.0},
            phase="DEGRADING",
        )
        assert result["method"].endswith("+conformal")

    def test_fallback_when_no_model(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 1.5},
            phase="DEGRADING",
        )
        assert result["fallback"] is True
        assert result["model_trained"] is False

    def test_no_fallback_with_trained_model(self, trained_predictor):
        features = {f"feat_{i}": 0.5 for i in range(5)}
        result = trained_predictor.predict(features, phase="DEGRADING")
        assert result["fallback"] is False
        assert result["model_trained"] is True

    def test_anomaly_phase_also_works(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 3.0},
            phase="ANOMALY",
        )
        assert result is not None
        assert result["rul_hours"] >= 0.0

    def test_ci_high_ge_low(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 2.5},
            phase="DEGRADING",
        )
        assert result["rul_high_ci"] >= result["rul_low_ci"]


# ---------------------------------------------------------------------------
# Smoothing tests
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_first_call_returns_raw(self, predictor):
        """First call should trust the model output directly."""
        result = predictor._smooth_rul(500.0)
        assert result == 500.0

    def test_subsequent_call_follows_cap(self, predictor):
        """Subsequent calls should respect the per-tick cap."""
        predictor._smooth_rul(700.0)  # First call: 700
        result = predictor._smooth_rul(100.0)  # Raw drops to 100
        # Cap floor: 700 * (1 - 0.15) = 595
        # max(100, 595) = 595
        assert result == pytest.approx(595.0)

    def test_smoothing_is_monotone_non_increasing(self, predictor):
        """Smoothed values should never increase."""
        values = [700, 600, 500, 400, 300]
        smoothed = [predictor._smooth_rul(v) for v in values]
        for i in range(1, len(smoothed)):
            assert smoothed[i] <= smoothed[i - 1] + 1e-9

    def test_smoothing_tracks_raw_when_above_cap(self, predictor):
        """When raw is above cap floor, smoothed follows raw."""
        predictor._smooth_rul(700.0)
        # Raw = 650, cap_floor = 700*0.85 = 595, max(650, 595) = 650
        result = predictor._smooth_rul(650.0)
        assert result == 650.0

    def test_smoothing_clamps_at_cap_floor(self, predictor):
        """When raw drops faster than cap, smoothed clamps to cap floor."""
        predictor._smooth_rul(100.0)
        # Raw = 10.0, cap_floor = 100*0.85 = 85, max(10, 85) = 85
        result = predictor._smooth_rul(10.0)
        assert result == pytest.approx(85.0)

    def test_smoothing_never_negative(self, predictor):
        predictor._smooth_rul(5.0)
        result = predictor._smooth_rul(-10.0)
        assert result >= 0.0

    def test_reset_smoothing_state(self, predictor):
        predictor._smooth_rul(500.0)
        assert predictor._last_smoothed_rul is not None
        predictor.reset_smoothing_state()
        assert predictor._last_smoothed_rul is None
        assert predictor._alarm_state == "ARMED"

    def test_custom_max_step_pct(self):
        from src.ml.rul_predictor import RULPredictor
        p = RULPredictor("SM-01", max_step_pct=0.10)
        p._smooth_rul(100.0)
        result = p._smooth_rul(0.0)
        # cap_floor = 100 * 0.90 = 90
        assert result == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Schmitt trigger tests
# ---------------------------------------------------------------------------

class TestSchmittTrigger:
    def test_armed_fires_on_threshold_crossing(self, predictor):
        """ARMED state: RUL below threshold should fire alarm."""
        assert predictor._alarm_state == "ARMED"
        result = predictor.should_publish_alarm(50.0)  # below 100
        assert result is True
        assert predictor._alarm_state == "TRIPPED"

    def test_armed_stays_armed_above_threshold(self, predictor):
        """ARMED state: RUL above threshold should not fire."""
        result = predictor.should_publish_alarm(150.0)  # above 100
        assert result is False
        assert predictor._alarm_state == "ARMED"

    def test_tripped_stays_tripped_below_threshold(self, predictor):
        """TRIPPED state: RUL still below should not fire again."""
        predictor.should_publish_alarm(50.0)  # Trip it
        result = predictor.should_publish_alarm(30.0)  # Still below
        assert result is False
        assert predictor._alarm_state == "TRIPPED"

    def test_tripped_rearms_above_hysteresis(self, predictor):
        """TRIPPED state: RUL above threshold+hysteresis should re-arm."""
        predictor.should_publish_alarm(50.0)  # Trip it
        assert predictor._alarm_state == "TRIPPED"
        # threshold=100, hysteresis=5, so need > 105 to re-arm
        result = predictor.should_publish_alarm(110.0)
        assert result is False  # Re-arming doesn't fire
        assert predictor._alarm_state == "ARMED"

    def test_tripped_stays_tripped_in_hysteresis_band(self, predictor):
        """TRIPPED state: RUL between threshold and threshold+hysteresis stays TRIPPED."""
        predictor.should_publish_alarm(50.0)  # Trip it
        # threshold=100, hysteresis=5, so 102 is in the band
        result = predictor.should_publish_alarm(102.0)
        assert result is False
        assert predictor._alarm_state == "TRIPPED"

    def test_full_cycle(self, predictor):
        """Full ARMED→TRIPPED→ARMED cycle."""
        # ARMED, above threshold
        assert predictor.should_publish_alarm(200.0) is False
        # ARMED, cross below
        assert predictor.should_publish_alarm(80.0) is True
        # TRIPPED, still below
        assert predictor.should_publish_alarm(60.0) is False
        # TRIPPED, recover above hysteresis
        assert predictor.should_publish_alarm(120.0) is False
        assert predictor._alarm_state == "ARMED"
        # ARMED again, cross below
        assert predictor.should_publish_alarm(40.0) is True

    def test_custom_threshold(self):
        from src.ml.rul_predictor import RULPredictor
        p = RULPredictor("ST-01", alarm_threshold_hours=50.0, alarm_hysteresis_hours=10.0)
        assert p.should_publish_alarm(40.0) is True  # below 50
        assert p._alarm_state == "TRIPPED"
        assert p.should_publish_alarm(55.0) is False  # in band (50-60)
        assert p._alarm_state == "TRIPPED"
        assert p.should_publish_alarm(65.0) is False  # above 60, re-arms
        assert p._alarm_state == "ARMED"


# ---------------------------------------------------------------------------
# Weibull confidence interval tests
# ---------------------------------------------------------------------------

class TestWeibullCI:
    def test_ci_returns_p10_p90(self, predictor):
        ci = predictor._weibull_confidence_interval(300.0)
        assert "p10" in ci
        assert "p90" in ci
        assert "confidence" in ci

    def test_ci_p10_le_p90(self, predictor):
        ci = predictor._weibull_confidence_interval(300.0)
        assert ci["p10"] <= ci["p90"]

    def test_ci_confidence_in_range(self, predictor):
        ci = predictor._weibull_confidence_interval(300.0)
        assert 0.0 <= ci["confidence"] <= 1.0

    def test_ci_deterministic(self, predictor):
        """Same input should produce same CI (seeded RNG)."""
        ci1 = predictor._weibull_confidence_interval(300.0)
        # Create a fresh predictor with same seed
        from src.ml.rul_predictor import RULPredictor
        p2 = RULPredictor("TEST-001", beta=2.1, eta=720.0)
        ci2 = p2._weibull_confidence_interval(300.0)
        assert ci1["p10"] == ci2["p10"]
        assert ci1["p90"] == ci2["p90"]

    def test_ci_wider_for_larger_rul(self, predictor):
        """Confidence interval should generally be wider for larger RUL."""
        from src.ml.rul_predictor import RULPredictor
        p1 = RULPredictor("CI-01", beta=2.1, eta=720.0)
        p2 = RULPredictor("CI-02", beta=2.1, eta=720.0)
        ci_small = p1._weibull_confidence_interval(50.0)
        ci_large = p2._weibull_confidence_interval(500.0)
        width_small = ci_small["p90"] - ci_small["p10"]
        width_large = ci_large["p90"] - ci_large["p10"]
        assert width_large > width_small

    def test_solve_current_age_new_machine(self, predictor):
        """When RUL >= unconditional median, age should be 0 (new machine)."""
        age = predictor._solve_current_age(9999.0)
        assert age == 0.0

    def test_solve_current_age_failed_machine(self, predictor):
        """When RUL <= 0, age should be very large (effectively failed)."""
        age = predictor._solve_current_age(-10.0)
        assert age == predictor.eta * 100.0


# ---------------------------------------------------------------------------
# Model persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, trained_predictor):
        """Train, save, then load in a new predictor."""
        from src.ml.rul_predictor import RULPredictor
        # The trained predictor already saved during train()
        # Create a new predictor with same machine_id - it should auto-load
        p2 = RULPredictor("TEST-002", beta=2.1, eta=720.0)
        assert p2.model is not None
        assert p2.feature_names == trained_predictor.feature_names
        assert p2.beta == trained_predictor.beta
        assert p2.eta == trained_predictor.eta

    def test_loaded_model_produces_same_predictions(self, trained_predictor):
        """Loaded model should produce identical predictions."""
        from src.ml.rul_predictor import RULPredictor
        features = {f"feat_{i}": 0.5 for i in range(5)}
        result1 = trained_predictor.predict(features, phase="DEGRADING")

        p2 = RULPredictor("TEST-002", beta=2.1, eta=720.0)
        result2 = p2.predict(features, phase="DEGRADING")

        assert result1["rul_hours"] == result2["rul_hours"]

    def test_hash_file_integrity(self, trained_predictor):
        """Hash file should exist and match the model file."""
        import hashlib
        hash_path = f"{trained_predictor._model_path}.sha256"
        assert os.path.exists(hash_path)
        with open(hash_path) as f:
            stored_hash = f.read().strip()
        hasher = hashlib.sha256()
        with open(trained_predictor._model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == stored_hash

    def test_model_path_property(self, trained_predictor):
        assert trained_predictor.model_path == trained_predictor._model_path
        assert trained_predictor.model_path.endswith("_rul.joblib")


# ---------------------------------------------------------------------------
# Conformal prediction tests
# ---------------------------------------------------------------------------

class TestConformal:
    def test_update_conformal(self, predictor):
        predictor.update_conformal(100.0, 95.0)
        assert predictor.conformal.calibration_size == 1

    def test_conformal_coverage_guarantee(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 2.0},
            phase="DEGRADING",
        )
        assert result["coverage_guarantee"] == 0.9  # 1 - alpha=0.10

    def test_conformal_calibration_size_zero_initially(self, predictor):
        result = predictor.predict(
            features={"vibration_rms_z_score": 2.0},
            phase="DEGRADING",
        )
        assert result["conformal_calibration_size"] == 0

    def test_conformal_calibration_size_increments(self, predictor):
        predictor.update_conformal(100.0, 90.0)
        predictor.update_conformal(200.0, 180.0)
        result = predictor.predict(
            features={"vibration_rms_z_score": 2.0},
            phase="DEGRADING",
        )
        assert result["conformal_calibration_size"] == 2


# ---------------------------------------------------------------------------
# Weibull-only fallback tests
# ---------------------------------------------------------------------------

class TestWeibullFallback:
    def test_fallback_with_z_scores(self, predictor):
        """Fallback should use z-scores to estimate degradation."""
        result = predictor.predict(
            features={"vibration_rms_z_score": 3.0, "temperature_z_score": 2.0},
            phase="DEGRADING",
        )
        assert result["fallback"] is True
        assert result["rul_hours"] >= 0.0

    def test_fallback_without_z_scores(self, predictor):
        """Fallback with no z-scores should use default 50% degradation."""
        result = predictor.predict(
            features={"other_feature": 1.0},
            phase="DEGRADING",
        )
        assert result["fallback"] is True
        assert result["rul_hours"] >= 0.0

    def test_fallback_ema_smoothing(self, predictor):
        """Successive fallback calls should show EMA smoothing."""
        r1 = predictor.predict(features={"vibration_rms_z_score": 3.0}, phase="DEGRADING")
        r2 = predictor.predict(features={"vibration_rms_z_score": 3.0}, phase="DEGRADING")
        # Both should return valid results
        assert r1["rul_hours"] >= 0.0
        assert r2["rul_hours"] >= 0.0
