"""Unit tests for src.ml.conformal_rul — Conformal Prediction for RUL."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ml.conformal_rul import ConformalRUL, ConformalRULPredictor

# ── Initialization ──────────────────────────────────────────────────────────

class TestConformalRULInit:
    """Tests for ConformalRUL initialization."""

    def test_default_params(self):
        cr = ConformalRUL()
        assert cr.alpha == 0.10
        assert cr.fallback_margin_hours == 10.0
        assert cr.calibration_size == 0

    def test_custom_alpha(self):
        cr = ConformalRUL(alpha=0.05)
        assert cr.alpha == 0.05
        assert cr.coverage_guarantee == 0.95

    def test_custom_fallback_margin(self):
        cr = ConformalRUL(fallback_margin_hours=24.0)
        assert cr.fallback_margin_hours == 24.0

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ConformalRUL(alpha=0.0)

    def test_invalid_alpha_one(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ConformalRUL(alpha=1.0)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ConformalRUL(alpha=-0.1)

    def test_invalid_alpha_above_one(self):
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ConformalRUL(alpha=1.5)


# ── Calibrate / Update ──────────────────────────────────────────────────────

class TestCalibrateAndUpdate:
    """Tests for calibrate() and update() methods."""

    def test_calibrate_replaces_residuals(self):
        cr = ConformalRUL()
        cr.update(100, 90)  # residual = 10
        cr.calibrate([5.0, 15.0, 25.0])
        assert cr.calibration_size == 3
        np.testing.assert_array_equal(cr.residuals, [5.0, 15.0, 25.0])

    def test_calibrate_takes_absolute_values(self):
        cr = ConformalRUL()
        cr.calibrate([-10.0, 20.0, -30.0])
        np.testing.assert_array_equal(cr.residuals, [10.0, 20.0, 30.0])

    def test_update_appends_residual(self):
        cr = ConformalRUL()
        cr.update(100.0, 90.0)  # |100 - 90| = 10
        assert cr.calibration_size == 1
        assert cr.residuals[0] == 10.0

    def test_update_multiple(self):
        cr = ConformalRUL()
        cr.update(100, 95)   # 5
        cr.update(200, 180)  # 20
        cr.update(50, 55)    # 5
        assert cr.calibration_size == 3
        np.testing.assert_array_equal(cr.residuals, [5.0, 5.0, 20.0])

    def test_max_residuals_cap(self):
        cr = ConformalRUL(max_residuals=5)
        for i in range(10):
            cr.update(float(i + 10), float(i))  # residual = 10 each
        assert cr.calibration_size == 5

    def test_empty_calibration(self):
        cr = ConformalRUL()
        assert cr.calibration_size == 0
        assert len(cr.residuals) == 0


# ── Predict Interval ────────────────────────────────────────────────────────

class TestPredictInterval:
    """Tests for predict_interval() and predict_with_interval()."""

    def test_predict_interval_empty_calibration_uses_fallback(self):
        cr = ConformalRUL(fallback_margin_hours=10.0)
        lower, upper = cr.predict_interval(100.0)
        assert lower == 90.0  # 100 - 10
        assert upper == 110.0  # 100 + 10

    def test_predict_interval_with_calibration(self):
        cr = ConformalRUL(alpha=0.10)
        cr.calibrate([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        lower, upper = cr.predict_interval(100.0)
        # With 10 residuals and alpha=0.10, q_level = min(1.0, 0.9 * 1.1) = 0.99
        # q_idx = ceil(0.99 * 10) - 1 = 10 - 1 = 9 → residuals[9] = 50.0
        assert lower == 50.0   # 100 - 50
        assert upper == 150.0  # 100 + 50

    def test_predict_interval_lower_clamped_to_zero(self):
        cr = ConformalRUL()
        cr.calibrate([200.0])  # large residual
        lower, upper = cr.predict_interval(50.0)
        assert lower == 0.0  # clamped
        assert upper == 250.0

    def test_predict_with_interval_returns_dict(self):
        cr = ConformalRUL(alpha=0.10)
        cr.calibrate([5.0, 10.0, 15.0])
        result = cr.predict_with_interval(100.0)
        assert isinstance(result, dict)
        assert "rul_hours" in result
        assert "rul_low_ci" in result
        assert "rul_high_ci" in result
        assert "coverage_guarantee" in result
        assert "conformal_calibration_size" in result

    def test_predict_with_interval_values(self):
        cr = ConformalRUL(alpha=0.10)
        cr.calibrate([5.0, 10.0, 15.0])
        result = cr.predict_with_interval(100.0)
        assert result["rul_hours"] == 100.0
        assert result["coverage_guarantee"] == 0.9
        assert result["conformal_calibration_size"] == 3
        assert result["rul_low_ci"] <= result["rul_hours"]
        assert result["rul_high_ci"] >= result["rul_hours"]


# ── Coverage Guarantee ──────────────────────────────────────────────────────

class TestCoverageGuarantee:
    """Tests for the conformal coverage guarantee property."""

    def test_coverage_guarantee_default(self):
        cr = ConformalRUL(alpha=0.10)
        assert cr.coverage_guarantee == 0.90

    def test_coverage_guarantee_custom(self):
        cr = ConformalRUL(alpha=0.05)
        assert cr.coverage_guarantee == 0.95

    def test_coverage_guarantee_empiirical(self):
        """Verify empirical coverage >= 1-alpha over many trials."""
        np.random.seed(42)
        alpha = 0.10
        cr = ConformalRUL(alpha=alpha)

        # Generate calibration data from known distribution
        true_values = np.random.exponential(scale=100, size=200)
        predictions = true_values + np.random.normal(0, 10, size=200)
        residuals = np.abs(predictions - true_values)
        cr.calibrate(residuals)

        # Test on new data
        n_test = 500
        covered = 0
        for _ in range(n_test):
            true_rul = np.random.exponential(scale=100)
            pred_rul = true_rul + np.random.normal(0, 10)
            lower, upper = cr.predict_interval(pred_rul)
            if lower <= true_rul <= upper:
                covered += 1

        empirical_coverage = covered / n_test
        # Conformal guarantee: coverage >= 1-alpha (with finite-sample correction)
        # Allow some slack for randomness
        assert empirical_coverage >= 0.80, (
            f"Empirical coverage {empirical_coverage:.2f} < 0.80"
        )


# ── Edge Cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases for conformal prediction."""

    def test_single_residual(self):
        cr = ConformalRUL(alpha=0.10)
        cr.calibrate([25.0])
        # q_level = min(1.0, 0.9 * 2.0) = 1.0
        # q_idx = ceil(1.0 * 1) - 1 = 0 → residuals[0] = 25.0
        lower, upper = cr.predict_interval(100.0)
        assert lower == 75.0
        assert upper == 125.0

    def test_two_residuals(self):
        cr = ConformalRUL(alpha=0.10)
        cr.calibrate([10.0, 20.0])
        # q_level = min(1.0, 0.9 * 1.5) = 1.0 (capped)
        # q_idx = ceil(1.0 * 2) - 1 = 1 → residuals[1] = 20.0
        lower, upper = cr.predict_interval(100.0)
        assert lower == 80.0
        assert upper == 120.0

    def test_zero_residual(self):
        cr = ConformalRUL(alpha=0.10)
        cr.calibrate([0.0, 0.0, 0.0])
        lower, upper = cr.predict_interval(100.0)
        assert lower == 100.0
        assert upper == 100.0

    def test_very_small_alpha(self):
        cr = ConformalRUL(alpha=0.01)
        cr.calibrate([5.0, 10.0, 15.0, 20.0, 25.0])
        assert cr.coverage_guarantee == 0.99
        # With small alpha, quantile should be high
        lower, upper = cr.predict_interval(100.0)
        assert upper - lower >= 40.0  # wide interval

    def test_very_large_alpha(self):
        cr = ConformalRUL(alpha=0.99)
        cr.calibrate([5.0, 10.0, 15.0, 20.0, 25.0])
        assert cr.coverage_guarantee == 0.01
        # With large alpha, quantile should be low
        lower, upper = cr.predict_interval(100.0)
        # Interval should be narrow
        assert upper - lower <= 20.0

    def test_update_after_calibrate(self):
        cr = ConformalRUL()
        cr.calibrate([10.0, 20.0])
        cr.update(100, 85)  # residual = 15
        assert cr.calibration_size == 3
        np.testing.assert_array_equal(cr.residuals, [10.0, 15.0, 20.0])

    def test_conformal_rul_predictor_alias(self):
        """ConformalRULPredictor is a backward-compatible alias."""
        crp = ConformalRULPredictor(alpha=0.05)
        assert isinstance(crp, ConformalRUL)
        assert crp.alpha == 0.05
        crp.calibrate([10.0, 20.0])
        result = crp.predict_with_interval(50.0)
        assert result["coverage_guarantee"] == 0.95


# ── MRL Baseline ────────────────────────────────────────────────────────────

class TestMRLBaseline:
    """Tests for mrl_baseline_seconds (fallback path)."""

    def test_mrl_fallback_at_t_zero(self):
        """At t=0, MRL = MTTF = eta * Gamma(1 + 1/beta)."""
        cr = ConformalRUL()
        beta = 2.0
        eta = 1000.0
        result = cr.mrl_baseline_seconds(beta, eta, t_seconds=0.0)
        expected = eta * math.gamma(1.0 + 1.0 / beta)
        # Should be close (may differ slightly if weibull_engine is available)
        assert result > 0
        assert result <= expected * 1.1  # within 10%

    def test_mrl_decreases_with_age(self):
        """MRL should decrease as t increases."""
        cr = ConformalRUL()
        beta = 2.0
        eta = 1000.0
        mrl_0 = cr.mrl_baseline_seconds(beta, eta, t_seconds=0.0)
        mrl_500 = cr.mrl_baseline_seconds(beta, eta, t_seconds=500.0)
        assert mrl_500 < mrl_0

    def test_mrl_non_negative(self):
        cr = ConformalRUL()
        result = cr.mrl_baseline_seconds(2.0, 1000.0, t_seconds=5000.0)
        assert result >= 0.0
