"""
Tests for WeibullParameters class — hazard_rate, survival_probability, mean_remaining_life.

Verifies the mathematical properties of the Weibull reliability model:
- Increasing hazard rate for beta > 1 (wear-out failure mode)
- Decreasing survival probability over time
- Decreasing mean remaining life as machine ages
- Boundary conditions at t=0
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_generator.weibull_engine import WeibullParameters


@pytest.fixture
def weibull_params():
    """Create a WeibullParameters instance with beta=2.0 (wear-out mode)."""
    rng = np.random.default_rng(42)
    return WeibullParameters(beta=2.0, eta=500000.0, rng=rng)


class TestHazardRate:
    def test_hazard_rate_increases_with_time(self, weibull_params):
        """h(10) < h(50) < h(90) for beta=2.0 (increasing hazard rate)."""
        h10 = weibull_params.hazard_rate(10.0)
        h50 = weibull_params.hazard_rate(50.0)
        h90 = weibull_params.hazard_rate(90.0)
        assert h10 < h50 < h90, (
            f"Hazard rate should increase with time for beta>1: "
            f"h(10)={h10}, h(50)={h50}, h(90)={h90}"
        )

    def test_hazard_rate_zero_at_t_zero(self, weibull_params):
        """h(0) == 0.0 for beta > 1."""
        h0 = weibull_params.hazard_rate(0.0)
        assert h0 == 0.0, f"Hazard rate at t=0 should be 0.0, got {h0}"

    def test_hazard_rate_positive_for_positive_t(self, weibull_params):
        """h(t) > 0 for t > 0 when beta > 1."""
        h = weibull_params.hazard_rate(1000.0)
        assert h > 0.0, f"Hazard rate should be positive for t>0, got {h}"


class TestSurvivalProbability:
    def test_survival_probability_decreases(self, weibull_params):
        """s(10) > s(50) > s(90) — survival probability decreases over time."""
        s10 = weibull_params.survival_probability(10.0)
        s50 = weibull_params.survival_probability(50.0)
        s90 = weibull_params.survival_probability(90.0)
        assert s10 > s50 > s90, (
            f"Survival probability should decrease: "
            f"s(10)={s10}, s(50)={s50}, s(90)={s90}"
        )

    def test_survival_probability_one_at_t_zero(self, weibull_params):
        """s(0) == 1.0 — certain survival at time zero."""
        s0 = weibull_params.survival_probability(0.0)
        assert s0 == 1.0, f"Survival probability at t=0 should be 1.0, got {s0}"

    def test_survival_probability_bounded(self, weibull_params):
        """s(t) is always in [0, 1]."""
        for t in [0, 100, 1000, 10000, 100000, 500000]:
            s = weibull_params.survival_probability(float(t))
            assert 0.0 <= s <= 1.0, f"s({t})={s} out of [0, 1] bounds"


class TestMeanRemainingLife:
    def test_mean_remaining_life_decreases(self, weibull_params):
        """mrl(0) > mrl(late) > mrl(very_late) — remaining life decreases with age.

        Use time values that are a significant fraction of eta (500000) to
        observe meaningful MRL decrease. At very early times (t << eta),
        MRL barely changes.
        """
        mrl0 = weibull_params.mean_remaining_life(0.0)
        mrl_mid = weibull_params.mean_remaining_life(200000.0)  # 40% of eta
        mrl_late = weibull_params.mean_remaining_life(400000.0)  # 80% of eta
        assert mrl0 > mrl_mid > mrl_late, (
            f"Mean remaining life should decrease: "
            f"mrl(0)={mrl0}, mrl(200k)={mrl_mid}, mrl(400k)={mrl_late}"
        )

    def test_mean_remaining_life_positive(self, weibull_params):
        """MRL is always non-negative."""
        for t in [0, 100, 1000, 10000]:
            mrl = weibull_params.mean_remaining_life(float(t))
            assert mrl >= 0.0, f"MRL at t={t} should be >= 0, got {mrl}"

    def test_mrl_at_zero_equals_mttf(self, weibull_params):
        """MRL(0) should equal MTTF = eta * Gamma(1 + 1/beta)."""
        mrl0 = weibull_params.mean_remaining_life(0.0)
        mttf = weibull_params.mean_time_to_failure()
        assert abs(mrl0 - mttf) < 1e-6, (
            f"MRL(0)={mrl0} should equal MTTF={mttf}"
        )
