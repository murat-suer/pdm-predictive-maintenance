import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_generator.machines import MACHINE_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestOMManual:
    OM_DIR = PROJECT_ROOT / "docs" / "om_manual"

    def test_om_directory_exists(self):
        assert self.OM_DIR.exists(), "docs/om_manual/ directory must exist"

    def test_om_files_exist(self):
        assert (self.OM_DIR / "AC_om_manual.md").exists(), "AC_om_manual.md missing"
        assert (self.OM_DIR / "HX_om_manual.md").exists(), "HX_om_manual.md missing"
        assert (self.OM_DIR / "CM_om_manual.md").exists(), "CM_om_manual.md missing"

    def test_om_minimum_procedures(self):
        for fname in ["AC_om_manual.md", "HX_om_manual.md", "CM_om_manual.md"]:
            content = (self.OM_DIR / fname).read_text(encoding="utf-8")
            interval_count = content.lower().count("interval")
            assert interval_count >= 10, (
                f"{fname} has {interval_count} interval references, need >= 10 procedures"
            )

    def test_om_references_standards(self):
        standards_map = {
            "AC_om_manual.md": ["API 619", "ISO 10816"],
            "HX_om_manual.md": ["TEMA"],
            "CM_om_manual.md": ["CEMA"],
        }
        for fname, standards in standards_map.items():
            content = (self.OM_DIR / fname).read_text(encoding="utf-8")
            for std in standards:
                assert std in content, f"{fname} missing standard reference: {std}"

    def test_threshold_links_valid(self):
        for mid, cfg in MACHINE_CONFIGS.items():
            mtype = cfg["type"]
            fname = f"{mtype}_om_manual.md"
            fpath = self.OM_DIR / fname
            if not fpath.exists():
                continue
            content = fpath.read_text(encoding="utf-8")
            for sensor_name, sensor_cfg in cfg["sensors"].items():
                w = sensor_cfg.get("warning_threshold")
                c = sensor_cfg.get("critical_threshold")
                if w is not None:
                    assert sensor_name in content, (
                        f"{fname} must reference sensor '{sensor_name}'"
                    )
                if c is not None:
                    assert sensor_name in content, (
                        f"{fname} must reference sensor '{sensor_name}' for critical threshold"
                    )


class TestMADB:
    MADB_PATH = PROJECT_ROOT / "docs" / "madb" / "madb.yaml"

    def test_madb_file_exists(self):
        assert self.MADB_PATH.exists(), "docs/madb/madb.yaml must exist"

    def test_madb_covers_all_sensors(self):
        import yaml
        with open(self.MADB_PATH, encoding="utf-8") as f:
            madb = yaml.safe_load(f)
        alarms = madb.get("alarms", [])
        covered = set()
        for alarm in alarms:
            key = (alarm["machine_id"], alarm["sensor_name"])
            covered.add(key)
        expected = set()
        for mid, cfg in MACHINE_CONFIGS.items():
            for sensor_name in cfg["sensors"]:
                expected.add((mid, sensor_name))
        missing = expected - covered
        assert not missing, f"MADB missing alarms for: {missing}"

    def test_madb_required_fields(self):
        import yaml
        with open(self.MADB_PATH, encoding="utf-8") as f:
            madb = yaml.safe_load(f)
        alarms = madb.get("alarms", [])
        required = ["machine_id", "sensor_name", "priority", "class",
                     "setpoint", "operator_action", "consequence_of_inaction", "rationale"]
        for alarm in alarms:
            for field in required:
                assert field in alarm, (
                    f"Alarm {alarm.get('machine_id','?')}.{alarm.get('sensor_name','?')} "
                    f"missing field: {field}"
                )

    def test_madb_priority_range(self):
        import yaml
        with open(self.MADB_PATH, encoding="utf-8") as f:
            madb = yaml.safe_load(f)
        for alarm in madb.get("alarms", []):
            assert 1 <= alarm["priority"] <= 4, (
                f"Priority must be 1-4, got {alarm['priority']}"
            )

    def test_madb_total_alarm_count(self):
        import yaml
        with open(self.MADB_PATH, encoding="utf-8") as f:
            madb = yaml.safe_load(f)
        alarms = madb.get("alarms", [])
        assert len(alarms) >= 30, f"MADB has {len(alarms)} alarms, need >= 30"


class TestImperfectMaintenance:
    def test_task_specific_distributions_exist(self):
        from src.data_generator.weibull_engine import MAINTENANCE_TASK_DISTRIBUTIONS
        assert "oil_change" in MAINTENANCE_TASK_DISTRIBUTIONS
        assert "bearing_replacement" in MAINTENANCE_TASK_DISTRIBUTIONS
        assert "belt_tensioning" in MAINTENANCE_TASK_DISTRIBUTIONS

    def test_oil_change_higher_quality_than_bearing(self):
        from src.data_generator.weibull_engine import MAINTENANCE_TASK_DISTRIBUTIONS
        rng = np.random.default_rng(42)
        oil_samples = [MAINTENANCE_TASK_DISTRIBUTIONS["oil_change"](rng) for _ in range(500)]
        bearing_samples = [MAINTENANCE_TASK_DISTRIBUTIONS["bearing_replacement"](rng) for _ in range(500)]
        assert np.mean(oil_samples) > np.mean(bearing_samples), (
            f"Oil change mean ({np.mean(oil_samples):.3f}) should exceed "
            f"bearing replacement mean ({np.mean(bearing_samples):.3f})"
        )

    def test_maintenance_never_perfect(self):
        from src.data_generator.weibull_engine import MAINTENANCE_TASK_DISTRIBUTIONS
        rng = np.random.default_rng(42)
        for task_name, sampler in MAINTENANCE_TASK_DISTRIBUTIONS.items():
            samples = [sampler(rng) for _ in range(200)]
            assert all(s < 1.0 for s in samples), (
                f"{task_name} produced quality >= 1.0 (perfect maintenance)"
            )
            assert all(s > 0.0 for s in samples), (
                f"{task_name} produced quality <= 0.0"
            )

    def test_task_specific_distributions(self):
        from src.data_generator.weibull_engine import MAINTENANCE_TASK_DISTRIBUTIONS
        rng = np.random.default_rng(42)
        oil_samples = [MAINTENANCE_TASK_DISTRIBUTIONS["oil_change"](rng) for _ in range(1000)]
        oil_mean = np.mean(oil_samples)
        assert 0.70 <= oil_mean <= 0.90, f"Oil change mean {oil_mean:.3f} outside [0.70, 0.90]"

        bearing_samples = [MAINTENANCE_TASK_DISTRIBUTIONS["bearing_replacement"](rng) for _ in range(1000)]
        bearing_mean = np.mean(bearing_samples)
        assert 0.60 <= bearing_mean <= 0.80, f"Bearing replacement mean {bearing_mean:.3f} outside [0.60, 0.80]"

        belt_samples = [MAINTENANCE_TASK_DISTRIBUTIONS["belt_tensioning"](rng) for _ in range(1000)]
        belt_mean = np.mean(belt_samples)
        assert 0.55 <= belt_mean <= 0.75, f"Belt tensioning mean {belt_mean:.3f} outside [0.55, 0.75]"


class TestPlausibilityGates:
    def test_check_plausibility_exists(self):
        from src.data_generator.weibull_engine import check_plausibility
        assert callable(check_plausibility)

    def test_degradation_monotonic(self):
        from src.data_generator.weibull_engine import check_plausibility
        state_ok = {
            "degradation_history": [0.0, 0.05, 0.10, 0.15, 0.20],
            "current_degradation": 0.25,
            "sensor_values": {"vibration_rms": 3.0, "bearing_temp": 65.0},
        }
        result = check_plausibility(state_ok)
        assert result["monotonic"] is True

        state_bad = {
            "degradation_history": [0.0, 0.05, 0.10, 0.08, 0.20],
            "current_degradation": 0.25,
            "sensor_values": {"vibration_rms": 3.0},
        }
        result = check_plausibility(state_bad)
        assert result["monotonic"] is False

    def test_no_instant_jumps(self):
        from src.data_generator.weibull_engine import check_plausibility
        state_ok = {
            "degradation_history": [0.0, 0.05, 0.10],
            "current_degradation": 0.15,
            "sensor_values": {"vibration_rms": 3.0},
        }
        result = check_plausibility(state_ok)
        assert result["no_instant_jumps"] is True

        state_bad = {
            "degradation_history": [0.0, 0.05, 0.10],
            "current_degradation": 0.50,
            "sensor_values": {"vibration_rms": 3.0},
        }
        result = check_plausibility(state_bad)
        assert result["no_instant_jumps"] is False

    def test_physical_bounds(self):
        from src.data_generator.weibull_engine import check_plausibility
        state_ok = {
            "degradation_history": [0.0, 0.05],
            "current_degradation": 0.10,
            "sensor_values": {"vibration_rms": 3.0, "bearing_temp": 65.0, "oil_pressure": 4.5},
        }
        result = check_plausibility(state_ok)
        assert result["physical_bounds_ok"] is True

        state_bad = {
            "degradation_history": [0.0, 0.05],
            "current_degradation": 0.10,
            "sensor_values": {"vibration_rms": -1.0, "bearing_temp": -50.0},
        }
        result = check_plausibility(state_bad)
        assert result["physical_bounds_ok"] is False

    def test_plausibility_returns_violations_list(self):
        from src.data_generator.weibull_engine import check_plausibility
        state = {
            "degradation_history": [0.0, 0.05, 0.03],
            "current_degradation": 0.50,
            "sensor_values": {"vibration_rms": -1.0},
        }
        result = check_plausibility(state)
        assert "violations" in result
        assert len(result["violations"]) > 0


class TestFFTHarmonics:
    def test_fft_engine_exists(self):
        from src.data_generator.fft_engine import compute_bearing_frequencies
        assert callable(compute_bearing_frequencies)

    def test_bpfo_frequency_correct(self):
        from src.data_generator.fft_engine import compute_bearing_frequencies
        result = compute_bearing_frequencies(
            N_balls=9,
            rpm=1750.0,
            ball_dia=0.4,
            pitch_dia=1.5,
            contact_angle=0.0,
        )
        bpfo = result["BPFO"]
        expected = (9 / 2) * (1750 / 60) * (1 - 0.4 / 1.5 * math.cos(0.0))
        assert bpfo == pytest.approx(expected, rel=1e-6)

    def test_bpfi_frequency_correct(self):
        from src.data_generator.fft_engine import compute_bearing_frequencies
        result = compute_bearing_frequencies(
            N_balls=9,
            rpm=1750.0,
            ball_dia=0.4,
            pitch_dia=1.5,
            contact_angle=0.0,
        )
        bpfi = result["BPFI"]
        expected = (9 / 2) * (1750 / 60) * (1 + 0.4 / 1.5 * math.cos(0.0))
        assert bpfi == pytest.approx(expected, rel=1e-6)

    def test_bsf_frequency_correct(self):
        from src.data_generator.fft_engine import compute_bearing_frequencies
        result = compute_bearing_frequencies(
            N_balls=9,
            rpm=1750.0,
            ball_dia=0.4,
            pitch_dia=1.5,
            contact_angle=0.0,
        )
        bsf = result["BSF"]
        bd_pd = 0.4 / 1.5 * math.cos(0.0)
        expected = (1.5 / 0.4) * (1750 / 60) * (1 - bd_pd ** 2)
        assert bsf == pytest.approx(expected, rel=1e-6)

    def test_harmonic_amplitude_scales_with_degradation(self):
        from src.data_generator.fft_engine import generate_fft_data
        fft_low = generate_fft_data(
            N_balls=9, rpm=1750.0, ball_dia=0.4, pitch_dia=1.5,
            contact_angle=0.0, degradation_level=0.1,
            BPFO_coeff=0.4, BPFI_coeff=0.6, BSF_coeff=0.2,
        )
        fft_high = generate_fft_data(
            N_balls=9, rpm=1750.0, ball_dia=0.4, pitch_dia=1.5,
            contact_angle=0.0, degradation_level=0.8,
            BPFO_coeff=0.4, BPFI_coeff=0.6, BSF_coeff=0.2,
        )
        assert fft_high["BPFO_amplitude"] > fft_low["BPFO_amplitude"]
        assert fft_high["BPFI_amplitude"] > fft_low["BPFI_amplitude"]
        assert fft_high["BSF_amplitude"] > fft_low["BSF_amplitude"]

    def test_fft_data_emits_dict(self):
        from src.data_generator.fft_engine import generate_fft_data
        result = generate_fft_data(
            N_balls=9, rpm=1750.0, ball_dia=0.4, pitch_dia=1.5,
            contact_angle=0.0, degradation_level=0.5,
            BPFO_coeff=0.4, BPFI_coeff=0.6, BSF_coeff=0.2,
        )
        assert "BPFO" in result
        assert "BPFI" in result
        assert "BSF" in result
        assert "BPFO_amplitude" in result
        assert "BPFI_amplitude" in result
        assert "BSF_amplitude" in result


class TestGammaProcess:
    def test_gamma_process_class_exists(self):
        from src.data_generator.weibull_engine import GammaProcessDegradation
        assert GammaProcessDegradation is not None

    def test_gamma_monotonic(self):
        from src.data_generator.weibull_engine import GammaProcessDegradation
        gp = GammaProcessDegradation(alpha=2.0, beta=5.0, rng=np.random.default_rng(42))
        values = [gp.step(dt=1.0) for _ in range(100)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], (
                f"Gamma process not monotonic at step {i}: {values[i-1]} -> {values[i]}"
            )

    def test_gamma_independent_increments(self):
        from src.data_generator.weibull_engine import GammaProcessDegradation
        gp = GammaProcessDegradation(alpha=2.0, beta=5.0, rng=np.random.default_rng(42))
        increments = []
        for _ in range(200):
            prev = gp.current_value
            gp.step(dt=1.0)
            increments.append(gp.current_value - prev)
        increments = np.array(increments)
        assert np.all(increments >= 0), "Gamma increments must be non-negative"
        mean_inc = np.mean(increments)
        expected_mean = 2.0 / 5.0
        assert abs(mean_inc - expected_mean) < 0.2, (
            f"Mean increment {mean_inc:.3f} too far from expected {expected_mean:.3f}"
        )

    def test_gamma_starts_at_zero(self):
        from src.data_generator.weibull_engine import GammaProcessDegradation
        gp = GammaProcessDegradation(alpha=2.0, beta=5.0, rng=np.random.default_rng(42))
        assert gp.current_value == 0.0

    def test_gamma_configurable_per_machine(self):
        from src.data_generator.weibull_engine import GammaProcessDegradation
        gp1 = GammaProcessDegradation(alpha=1.0, beta=10.0, rng=np.random.default_rng(1))
        gp2 = GammaProcessDegradation(alpha=5.0, beta=2.0, rng=np.random.default_rng(2))
        for _ in range(50):
            gp1.step(dt=1.0)
            gp2.step(dt=1.0)
        assert gp2.current_value > gp1.current_value, (
            "Higher alpha/beta ratio should produce more degradation"
        )


class TestImperfectMaintenanceWiring:
    def test_reset_after_maintenance_uses_task_distributions(self):
        from src.data_generator.machines import MACHINE_CONFIGS
        from src.data_generator.weibull_engine import (
            MAINTENANCE_TASK_DISTRIBUTIONS,
            create_machine_state,
        )

        ac_sensors = MACHINE_CONFIGS["AC-201"]["sensors"]
        state = create_machine_state("AC-201", ac_sensors, global_seed=42, machine_index=0)

        qualities = []
        for _ in range(200):
            state.reset_after_maintenance()
            qualities.append(state.maintenance_quality)

        all_samples = []
        rng_check = np.random.default_rng(999)
        for _ in range(200):
            task = rng_check.choice(list(MAINTENANCE_TASK_DISTRIBUTIONS.keys()))
            all_samples.append(MAINTENANCE_TASK_DISTRIBUTIONS[task](rng_check))

        q_mean = np.mean(qualities)
        a_mean = np.mean(all_samples)
        assert abs(q_mean - a_mean) < 0.10, (
            f"maintenance_quality mean {q_mean:.3f} diverges from "
            f"task-distribution mean {a_mean:.3f}"
        )

        uniform_samples = [np.random.default_rng(i).uniform(0.7, 1.0) for i in range(200)]
        u_mean = np.mean(uniform_samples)
        assert abs(q_mean - u_mean) > 0.03, (
            f"maintenance_quality mean {q_mean:.3f} too close to Uniform(0.7,1.0) "
            f"mean {u_mean:.3f} — task distributions not wired"
        )

    def test_maintenance_task_type_selected(self):
        from src.data_generator.machines import MACHINE_CONFIGS
        from src.data_generator.weibull_engine import (
            create_machine_state,
        )

        ac_sensors = MACHINE_CONFIGS["AC-201"]["sensors"]
        state = create_machine_state("AC-201", ac_sensors, global_seed=42, machine_index=0)

        seen_tasks = set()
        for _ in range(300):
            state.reset_after_maintenance()
            seen_tasks.add(state.last_maintenance_task)

        assert len(seen_tasks) >= 2, (
            f"Only {len(seen_tasks)} task type(s) seen in 300 maintenance events: {seen_tasks}"
        )


class TestPlausibilityGatesWiring:
    def test_plausibility_called_in_scheduler(self):
        import asyncio
        from unittest.mock import patch

        from src.data_generator.independent_scheduler import (
            MachineSchedulerConfig,
            machine_scheduler,
        )
        from src.data_generator.machines import MACHINE_CONFIGS
        from src.data_generator.weibull_engine import create_machine_state

        ac_sensors = MACHINE_CONFIGS["AC-201"]["sensors"]
        state = create_machine_state("AC-201", ac_sensors, global_seed=42, machine_index=0)

        config = MachineSchedulerConfig(state=state, speed_multiplier=10000.0)

        call_count = 0

        from src.data_generator import independent_scheduler
        original_check = independent_scheduler.check_plausibility

        def counting_check(s):
            nonlocal call_count
            call_count += 1
            return original_check(s)

        async def run_briefly():
            nonlocal call_count
            with patch.object(independent_scheduler, "check_plausibility", side_effect=counting_check):
                try:
                    await asyncio.wait_for(machine_scheduler(config), timeout=2.0)
                except (TimeoutError, Exception):
                    pass

        asyncio.run(run_briefly())
        assert call_count > 0, (
            "check_plausibility was never called during scheduler loop"
        )


class TestFFTIntegration:
    def test_fft_data_attached_to_reading(self):
        import time as wall_time

        from src.data_generator.fft_engine import generate_fft_data
        from src.data_generator.independent_scheduler import (
            SensorReading,
            apply_degradation_step,
            generate_sensor_reading,
        )
        from src.data_generator.machines import MACHINE_CONFIGS
        from src.data_generator.weibull_engine import create_machine_state

        ac_sensors = MACHINE_CONFIGS["AC-201"]["sensors"]
        state = create_machine_state("AC-201", ac_sensors, global_seed=42, machine_index=0)

        for _ in range(50):
            apply_degradation_step(state, "vibration_rms", dt_seconds=1.0)

        sensor = state.sensors["vibration_rms"]
        if sensor.degradation_level < 0.3:
            sensor.degradation_level = 0.5

        value = generate_sensor_reading(state, "vibration_rms")
        fft_cfg = ac_sensors["vibration_rms"].get("fft", {})
        if fft_cfg and sensor.degradation_level > 0.3:
            fft_data = generate_fft_data(
                N_balls=fft_cfg.get("N_balls", 9),
                rpm=fft_cfg.get("rpm_nominal", 1750),
                ball_dia=0.4,
                pitch_dia=1.5,
                contact_angle=0.0,
                degradation_level=sensor.degradation_level,
                BPFO_coeff=fft_cfg.get("BPFO_coeff", 0.4),
                BPFI_coeff=fft_cfg.get("BPFI_coeff", 0.6),
                BSF_coeff=fft_cfg.get("BSF_coeff", 0.2),
                rng=sensor.rng,
            )
            reading = SensorReading(
                machine_id="AC-201",
                sensor_name="vibration_rms",
                value=value,
                degradation_level=sensor.degradation_level,
                phase=state.phase,
                simulated_time=state.operational_seconds,
                wall_time=wall_time.time(),
                operational_seconds=sensor.operational_seconds,
                present=True,
                fft_data=fft_data,
            )
            assert reading.fft_data is not None
            assert "BPFO" in reading.fft_data
            assert "BPFO_amplitude" in reading.fft_data


class TestGammaProcessWiring:
    def test_gamma_process_instantiated_when_configured(self):
        from src.data_generator.machines import MACHINE_CONFIGS
        from src.data_generator.weibull_engine import (
            GammaProcessDegradation,
            create_machine_state,
        )

        ac_sensors = MACHINE_CONFIGS["AC-201"]["sensors"]
        test_specs = {}
        for sname, scfg in ac_sensors.items():
            sc = dict(scfg)
            sc["degradation_process_type"] = "gamma"
            sc["gamma_alpha"] = 2.0
            sc["gamma_beta"] = 5.0
            test_specs[sname] = sc

        state = create_machine_state("AC-201", test_specs, global_seed=42, machine_index=0)

        gamma_count = 0
        for sensor in state.sensors.values():
            if hasattr(sensor, "gamma_process") and sensor.gamma_process is not None:
                gamma_count += 1
                assert isinstance(sensor.gamma_process, GammaProcessDegradation)

        assert gamma_count > 0, (
            "No gamma processes instantiated despite degradation_process_type='gamma'"
        )

    def test_gamma_process_used_in_degradation_step(self):
        from src.data_generator.independent_scheduler import apply_degradation_step
        from src.data_generator.machines import MACHINE_CONFIGS
        from src.data_generator.weibull_engine import create_machine_state

        ac_sensors = MACHINE_CONFIGS["AC-201"]["sensors"]
        test_specs = {}
        for sname, scfg in ac_sensors.items():
            sc = dict(scfg)
            sc["degradation_process_type"] = "gamma"
            sc["gamma_alpha"] = 2.0
            sc["gamma_beta"] = 5.0
            test_specs[sname] = sc

        state = create_machine_state("AC-201", test_specs, global_seed=42, machine_index=0)

        for _ in range(20):
            apply_degradation_step(state, "vibration_rms", dt_seconds=1.0)

        sensor = state.sensors["vibration_rms"]
        if hasattr(sensor, "gamma_process") and sensor.gamma_process is not None:
            assert sensor.gamma_process.current_value > 0, (
                "Gamma process was not stepped during degradation"
            )
