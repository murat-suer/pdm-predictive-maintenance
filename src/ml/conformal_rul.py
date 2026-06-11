"""Streaming conformal prediction intervals for RUL point estimates."""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class ConformalRUL:
    """Distribution-free RUL intervals calibrated from realized failures."""

    def __init__(self, alpha: float = 0.10, max_residuals: int = 1000, fallback_margin_hours: float = 10.0):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.fallback_margin_hours = fallback_margin_hours
        self._residuals = deque(maxlen=max_residuals)

    @property
    def residuals(self) -> np.ndarray:
        """Sorted absolute residuals used for split conformal quantiles."""
        return np.sort(np.asarray(self._residuals, dtype=float))

    @property
    def calibration_size(self) -> int:
        return len(self._residuals)

    @property
    def coverage_guarantee(self) -> float:
        return round(1.0 - self.alpha, 2)

    def calibrate(self, residuals) -> None:
        """Replace calibration residuals with absolute held-out errors."""
        self._residuals.clear()
        for residual in residuals:
            self._residuals.append(abs(float(residual)))

    def update(self, predicted_hours: float, actual_hours: float) -> None:
        """Append one realized absolute error from a failure outcome."""
        self._residuals.append(abs(float(predicted_hours) - float(actual_hours)))

    def predict_interval(self, rul_point: float) -> tuple[float, float]:
        """Return the conformal P10/P90 interval around one point estimate."""
        q_hat = self._quantile()
        return max(0.0, rul_point - q_hat), max(0.0, rul_point + q_hat)

    def predict_with_interval(self, rul_point: float) -> dict:
        """Return a point estimate with interval metadata for API payloads."""
        lower, upper = self.predict_interval(rul_point)
        return {
            "rul_hours": rul_point,
            "rul_low_ci": lower,
            "rul_high_ci": upper,
            "coverage_guarantee": self.coverage_guarantee,
            "conformal_calibration_size": self.calibration_size,
        }

    def _quantile(self) -> float:
        residuals = self.residuals
        n = len(residuals)
        if n == 0:
            return self.fallback_margin_hours
        q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        q_idx = min(int(np.ceil(q_level * n)) - 1, n - 1)
        return float(residuals[max(0, q_idx)])

    def mrl_baseline_seconds(
        self,
        beta: float,
        eta_seconds: float,
        t_seconds: float = 0.0,
    ) -> float:
        """
        Phase 4 Item 2 — closed-form Mean Remaining Life from a Weibull
        prior, used as the conformal predictor's calibration target.

        For a sensor/machine whose reliability follows Weibull(beta, eta),
        the MRL at age t is:

            MRL(t) = (η/β) · exp((t/η)^β) · Γ(1/β, (t/η)^β)

        Implemented via scipy.special.gammaincc (regularized upper
        incomplete Γ) composed with math.gamma (Γ(a) factor).

        Reference: Lai & Xie 2006, "Stochastic Ageing and Dependence
        for Reliability", Springer, §2.3 Eq. 2.17.

        This is a thin pass-through to the closed-form MRL on
        `WeibullParameters` so the conformal module can compare its
        empirical RUL interval against an analytical ground truth.
        A residual significantly larger than the conformal quantile
        is a signal that the empirical RUL estimate is diverging
        from the simulator's Weibull prior — useful for drift
        detection and for tuning the alpha target.

        Parameters
        ----------
        beta
            Weibull shape (1.5-2.5 for wear-out mode).
        eta_seconds
            Weibull scale in seconds (characteristic life).
        t_seconds
            Current age in seconds. Default 0 → returns MTTF.

        Returns
        -------
        float
            MRL in seconds. 0 if survival is essentially 0.
        """
        try:
            from src.data_generator.weibull_engine import WeibullParameters

            wp = WeibullParameters(
                beta=float(beta),
                eta=float(eta_seconds),
                rng=np.random.default_rng(0),
            )
            return float(wp.mean_remaining_life(float(t_seconds)))
        except (ImportError, ValueError, OverflowError):
            # Fallback when scipy or the engine is unavailable —
            # return a coarse estimate from the closed-form MTTF
            # minus the current age.
            try:
                mttf = float(eta_seconds) * float(
                    math.gamma(1.0 + 1.0 / float(beta))
                )
            except (ValueError, OverflowError):
                mttf = float(eta_seconds)
            return float(max(0.0, mttf - float(t_seconds)))


class ConformalRULPredictor(ConformalRUL):
    """Backward-compatible alias for older imports."""
