"""
Tests for sensor degradation model subclasses:
- BearingISODegradation (ISO 281 L10 bearing life)
- OilArrheniusDegradation (Arrhenius oil degradation)
- FoulingTEMADegradation (TEMA RGP-T-2.4 fouling)

Verifies that degradation increases monotonically with time for each model.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_generator.weibull_engine import (
    BearingISODegradation,
    DegradationModelType,
    FoulingTEMADegradation,
    OilArrheniusDegradation,
    SensorDegradationConfig,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestBearingISODegradation:
    def test_bearing_iso_degradation_increases(self, rng):
        """Bearing degradation should increase over time."""
        config = SensorDegradationConfig(
            model_type=DegradationModelType.BEARING_ISO281,
            nominal_mu=1.0,
            nominal_sigma=0.1,
            degradation_direction=1,
            degradation_weight=1.0,
            model_params={"L10_hours": 720.0, "load_ratio": 1.0, "bearing_type": "ball"},
        )
        model = BearingISODegradation(config)

        d_100h = model._compute_degradation(100 * 3600)
        d_500h = model._compute_degradation(500 * 3600)
        d_1000h = model._compute_degradation(1000 * 3600)

        assert d_100h < d_500h < d_1000h, (
            f"Bearing degradation should increase: "
            f"d(100h)={d_100h:.4f}, d(500h)={d_500h:.4f}, d(1000h)={d_1000h:.4f}"
        )

    def test_bearing_iso_starts_at_zero(self, rng):
        """Bearing degradation at t=0 should be 0."""
        config = SensorDegradationConfig(
            model_type=DegradationModelType.BEARING_ISO281,
            nominal_mu=1.0,
            nominal_sigma=0.1,
            degradation_direction=1,
            degradation_weight=1.0,
            model_params={"L10_hours": 720.0},
        )
        model = BearingISODegradation(config)
        d0 = model._compute_degradation(0.0)
        assert d0 == 0.0, f"Degrade at t=0 should be 0.0, got {d0}"


class TestOilArrheniusDegradation:
    def test_oil_arrhenius_degradation_increases(self, rng):
        """Oil degradation should increase over time (Arrhenius model)."""
        config = SensorDegradationConfig(
            model_type=DegradationModelType.OIL_ARRHENIUS,
            nominal_mu=5.0,
            nominal_sigma=0.5,
            degradation_direction=-1,
            degradation_weight=1.0,
            model_params={
                "T_operating_K": 353.15,  # 80°C
                "T_reference_K": 373.15,  # 100°C
                "Ea_J_per_mol": 85000.0,
                "k_ref_per_hour": 1.0 / 5000.0,
            },
        )
        model = OilArrheniusDegradation(config)

        d_100h = model._compute_degradation(100 * 3600)
        d_1000h = model._compute_degradation(1000 * 3600)
        d_5000h = model._compute_degradation(5000 * 3600)

        assert d_100h < d_1000h < d_5000h, (
            f"Oil degradation should increase: "
            f"d(100h)={d_100h:.4f}, d(1000h)={d_1000h:.4f}, d(5000h)={d_5000h:.4f}"
        )

    def test_oil_arrhenius_starts_at_zero(self, rng):
        """Oil degradation at t=0 should be 0."""
        config = SensorDegradationConfig(
            model_type=DegradationModelType.OIL_ARRHENIUS,
            nominal_mu=5.0,
            nominal_sigma=0.5,
            degradation_direction=-1,
            degradation_weight=1.0,
            model_params={},
        )
        model = OilArrheniusDegradation(config)
        d0 = model._compute_degradation(0.0)
        assert d0 == 0.0, f"Oil degrade at t=0 should be 0.0, got {d0}"


class TestFoulingTEMADegradation:
    def test_fouling_tema_degradation_increases(self, rng):
        """Fouling degradation should increase over time (TEMA model)."""
        config = SensorDegradationConfig(
            model_type=DegradationModelType.FOULING_TEMA,
            nominal_mu=10.0,
            nominal_sigma=1.0,
            degradation_direction=1,
            degradation_weight=1.0,
            model_params={
                "Rf_max": 0.00035,
                "k_foul_per_hour": 0.001,
            },
        )
        model = FoulingTEMADegradation(config)

        d_100h = model._compute_degradation(100 * 3600)
        d_1000h = model._compute_degradation(1000 * 3600)
        d_5000h = model._compute_degradation(5000 * 3600)

        assert d_100h < d_1000h < d_5000h, (
            f"Fouling degradation should increase: "
            f"d(100h)={d_100h:.4f}, d(1000h)={d_1000h:.4f}, d(5000h)={d_5000h:.4f}"
        )

    def test_fouling_tema_starts_at_zero(self, rng):
        """Fouling degradation at t=0 should be 0."""
        config = SensorDegradationConfig(
            model_type=DegradationModelType.FOULING_TEMA,
            nominal_mu=10.0,
            nominal_sigma=1.0,
            degradation_direction=1,
            degradation_weight=1.0,
            model_params={},
        )
        model = FoulingTEMADegradation(config)
        d0 = model._compute_degradation(0.0)
        assert d0 == 0.0, f"Fouling degrade at t=0 should be 0.0, got {d0}"

    def test_fouling_tema_bounded(self, rng):
        """Fouling degradation should be bounded [0, 1]."""
        config = SensorDegradationConfig(
            model_type=DegradationModelType.FOULING_TEMA,
            nominal_mu=10.0,
            nominal_sigma=1.0,
            degradation_direction=1,
            degradation_weight=1.0,
            model_params={"Rf_max": 0.00035, "k_foul_per_hour": 0.001},
        )
        model = FoulingTEMADegradation(config)
        # Very large time should approach 1.0 but not exceed it
        d_large = model._compute_degradation(100000 * 3600)
        assert 0.0 <= d_large <= 1.0, f"Fouling degrade should be in [0,1], got {d_large}"
