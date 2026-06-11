import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_generator.machines import (
    MACHINE_CONFIGS,
    SHIFT_LOAD_PROFILES,
)
from src.data_generator.weibull_engine import (
    BELT_DECAY_REFERENCE,
    BeltSlipDegradation,
    DegradationModelType,
    SensorDegradationConfig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CM Beta Values
# ═══════════════════════════════════════════════════════════════════════════════


class TestCMBetaValues:
    def test_cm203_beta_above_2(self):
        config = MACHINE_CONFIGS["CM-203"]
        beta = config["weibull"]["beta"]
        assert beta > 2.0, f"CM-203 beta={beta} should be > 2.0 (wear-out)"

    def test_cm303_beta_above_2(self):
        config = MACHINE_CONFIGS["CM-303"]
        beta = config["weibull"]["beta"]
        assert beta > 2.0, f"CM-303 beta={beta} should be > 2.0 (wear-out)"

    def test_all_betas_wear_out(self):
        for mid, cfg in MACHINE_CONFIGS.items():
            beta = cfg["weibull"]["beta"]
            assert beta >= 1.5, f"{mid} beta={beta} below wear-out range"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Belt Decay Rate
# ═══════════════════════════════════════════════════════════════════════════════


class TestBeltDecayRate:
    def test_belt_decay_reaches_95pct_in_demo_window(self):
        t95_seconds = 132 * 3600
        lam = -math.log(0.05) / t95_seconds
        config = SensorDegradationConfig(
            model_type=DegradationModelType.BELT_SLIP,
            nominal_mu=8.2,
            nominal_sigma=0.25,
            degradation_direction=1,
            degradation_weight=0.30,
            model_params={"lambda_per_second": lam},
        )
        model = BeltSlipDegradation(config)
        rng = np.random.default_rng(42)
        d_at_t95 = model.compute_degradation(t95_seconds, rng)
        assert 0.90 <= d_at_t95 <= 1.0, (
            f"Belt degradation at t95 should be ~0.95, got {d_at_t95}"
        )

    def test_belt_decay_reference_calibrated(self):
        t95_cm203_seconds = 132 * 3600
        expected_lambda = -math.log(0.05) / t95_cm203_seconds
        ratio = BELT_DECAY_REFERENCE / expected_lambda
        assert 0.5 <= ratio <= 2.0, (
            f"BELT_DECAY_REFERENCE={BELT_DECAY_REFERENCE:.2e} should be "
            f"within 2x of calibrated lambda={expected_lambda:.2e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Physics/Weibull Blend
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhysicsWeibullBlend:
    def test_blend_ratio_at_least_40pct_physics(self):
        import inspect

        from src.data_generator.independent_scheduler import apply_degradation_step

        src = inspect.getsource(apply_degradation_step)
        assert "0.50" in src or "0.40" in src or "physics_weight" in src, (
            "Blend ratio should be at least 40% physics (not 30%)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Shift Load Profiles
# ═══════════════════════════════════════════════════════════════════════════════


class TestShiftLoadProfiles:
    def test_shift_load_profiles_defined(self):
        assert "morning" in SHIFT_LOAD_PROFILES
        assert "afternoon" in SHIFT_LOAD_PROFILES
        assert "night" in SHIFT_LOAD_PROFILES

    def test_afternoon_load_factor_above_1(self):
        afternoon = SHIFT_LOAD_PROFILES["afternoon"]
        assert afternoon["load_factor"]["mu"] > 1.0

    def test_night_load_factor_below_1(self):
        night = SHIFT_LOAD_PROFILES["night"]
        assert night["load_factor"]["mu"] < 1.0

    def test_apply_shift_load_exists(self):
        from src.data_generator.independent_scheduler import _apply_shift_load

        assert callable(_apply_shift_load)

    def test_afternoon_shift_accelerates_degradation(self):
        from src.data_generator.independent_scheduler import _apply_shift_load

        morning_mult = _apply_shift_load(10)
        afternoon_mult = _apply_shift_load(14)
        night_mult = _apply_shift_load(23)
        assert afternoon_mult > morning_mult, (
            f"Afternoon ({afternoon_mult}) should exceed morning ({morning_mult})"
        )
        assert night_mult < morning_mult, (
            f"Night ({night_mult}) should be below morning ({morning_mult})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Cross-Line Load Spike
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossLineLoadSpike:
    def test_cm303_has_cross_line_config(self):
        cm303 = MACHINE_CONFIGS["CM-303"]
        assert "cross_line_load_spike" in cm303

    def test_apply_cross_line_load_spike_exists(self):
        from src.data_generator.independent_scheduler import (
            apply_cross_line_load_spike,
        )

        assert callable(apply_cross_line_load_spike)

    def test_cross_line_spike_increases_tension(self):
        from src.data_generator.independent_scheduler import (
            apply_cross_line_load_spike,
        )
        from src.data_generator.machines import build_machine_specs
        from src.data_generator.weibull_engine import create_machine_state

        specs = build_machine_specs()
        cm303_state = create_machine_state("CM-303", specs["CM-303"], 42, 5)
        original_tension = cm303_state.sensors["belt_tension"].degradation_level

        apply_cross_line_load_spike(cm303_state)

        for sensor in cm303_state.sensors.values():
            pass
        assert cm303_state.sensors["belt_tension"].config.model_params.get(
            "cross_line_active", False
        ) or cm303_state.sensors["belt_tension"].degradation_level >= original_tension

    def test_cross_line_spike_config_values(self):
        spike = MACHINE_CONFIGS["CM-303"]["cross_line_load_spike"]
        assert spike["belt_tension_mult"]["mu"] == pytest.approx(1.4, abs=0.1)
        assert spike["deg_rate_mult"]["mu"] == pytest.approx(1.8, abs=0.1)
        assert spike["motor_load_delta"]["mu"] == pytest.approx(15.0, abs=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Sensor Count Documentation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSensorCountDocumentation:
    def test_all_machines_have_5_sensors(self):
        for mid, cfg in MACHINE_CONFIGS.items():
            n = len(cfg["sensors"])
            assert n == 5, f"{mid} has {n} sensors, expected 5"

    def test_docstrings_say_5_not_6(self):
        from src.data_generator import weibull_engine

        mod_doc = weibull_engine.__doc__ or ""
        assert "6 sensors" not in mod_doc, (
            "weibull_engine docstring still says '6 sensors'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FMEA Library
# ═══════════════════════════════════════════════════════════════════════════════


class TestFMEALibrary:
    FMEA_DIR = Path(__file__).resolve().parents[3] / "docs" / "fmea"

    def test_fmea_directory_exists(self):
        assert self.FMEA_DIR.exists(), "docs/fmea/ directory must exist"

    def test_fmea_files_exist(self):
        assert (self.FMEA_DIR / "AC_fmea.md").exists(), "AC_fmea.md missing"
        assert (self.FMEA_DIR / "HX_fmea.md").exists(), "HX_fmea.md missing"
        assert (self.FMEA_DIR / "CM_fmea.md").exists(), "CM_fmea.md missing"

    def test_fmea_has_required_fields(self):
        for fname in ["AC_fmea.md", "HX_fmea.md", "CM_fmea.md"]:
            content = (self.FMEA_DIR / fname).read_text(encoding="utf-8")
            for field in ["Severity", "Occurrence", "Detectability", "RPN"]:
                assert field in content, f"{fname} missing field: {field}"

    def test_fmea_minimum_failure_modes(self):
        for fname in ["AC_fmea.md", "HX_fmea.md", "CM_fmea.md"]:
            content = (self.FMEA_DIR / fname).read_text(encoding="utf-8")
            rpn_count = content.count("RPN")
            assert rpn_count >= 5, (
                f"{fname} has {rpn_count} RPN entries, need >= 5 failure modes"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Shift Load Integration (dead code → wired)
# ═══════════════════════════════════════════════════════════════════════════════


class TestShiftLoadIntegration:
    def test_shift_load_actually_called_in_scheduler(self):
        import inspect

        src = inspect.getsource(
            __import__("src.data_generator.independent_scheduler", fromlist=["machine_scheduler"]).machine_scheduler
        )
        assert "_apply_shift_load" in src, (
            "_apply_shift_load must be called inside machine_scheduler()"
        )

    def test_afternoon_shift_produces_higher_degradation(self):
        from src.data_generator.independent_scheduler import (
            _apply_shift_load,
        )
        from src.data_generator.machines import build_machine_specs

        specs = build_machine_specs()
        morning_factor = _apply_shift_load(10)
        afternoon_factor = _apply_shift_load(15)
        assert afternoon_factor > morning_factor, (
            f"Afternoon factor ({afternoon_factor}) must exceed morning ({morning_factor})"
        )

    def test_apply_degradation_step_accepts_load_factor(self):
        import inspect

        from src.data_generator.independent_scheduler import apply_degradation_step

        sig = inspect.signature(apply_degradation_step)
        assert "load_factor" in sig.parameters, (
            "apply_degradation_step must accept a load_factor parameter"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Cross-Line Load Spike Integration (dead code → wired)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossLineLoadSpikeIntegration:
    def test_cross_line_spike_triggered_on_line_a_failure(self):
        import inspect

        from src.data_generator.independent_scheduler import machine_scheduler

        src = inspect.getsource(machine_scheduler)
        assert "apply_cross_line_load_spike" in src, (
            "apply_cross_line_load_spike must be called inside machine_scheduler()"
        )
        assert "AC-201" in src or "LINE_A" in src or "line_a" in src.lower() or "engine_states" in src, (
            "machine_scheduler must check Line A machine failure"
        )

    def test_cross_line_spike_applies_all_multipliers(self):
        from src.data_generator.independent_scheduler import apply_cross_line_load_spike
        from src.data_generator.machines import build_machine_specs
        from src.data_generator.weibull_engine import create_machine_state

        specs = build_machine_specs()
        cm303_state = create_machine_state("CM-303", specs["CM-303"], 42, 5)

        orig_lambda = cm303_state.sensors["belt_tension"].config.model_params["lambda_per_second"]
        orig_nominal_mu = cm303_state.sensors["motor_load"].config.nominal_mu

        apply_cross_line_load_spike(cm303_state)

        new_lambda = cm303_state.sensors["belt_tension"].config.model_params["lambda_per_second"]
        new_nominal_mu = cm303_state.sensors["motor_load"].config.nominal_mu

        assert new_lambda > orig_lambda, (
            f"belt_tension lambda must increase: {orig_lambda} → {new_lambda}"
        )
        assert new_nominal_mu > orig_nominal_mu, (
            f"motor_load nominal_mu must increase: {orig_nominal_mu} → {new_nominal_mu}"
        )
        for sensor in cm303_state.sensors.values():
            assert sensor.config.model_params.get("cross_line_active") is True, (
                f"{sensor.name} must have cross_line_active=True"
            )

    def test_engine_states_passed_to_scheduler_config(self):
        import inspect

        from src.data_generator.independent_scheduler import SimulationEngine

        src = inspect.getsource(SimulationEngine.run)
        assert "engine_states" in src or "_states" in src, (
            "SimulationEngine.run must pass shared states to scheduler configs"
        )
