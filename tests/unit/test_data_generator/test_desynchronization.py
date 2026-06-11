import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data_generator.independent_scheduler import apply_degradation_step
from src.data_generator.machines import build_machine_specs
from src.data_generator.weibull_engine import (
    create_machine_state,
)

IDENTICAL_PAIRS = [
    ("AC-201", "AC-301"),
    ("HX-202", "HX-302"),
    ("CM-203", "CM-303"),
]


def _make_pair(pair, seed=42):
    specs = build_machine_specs()
    m_a = pair[0]
    m_b = pair[1]
    idx_a = list(build_machine_specs().keys()).index(m_a)
    idx_b = list(build_machine_specs().keys()).index(m_b)
    state_a = create_machine_state(m_a, specs[m_a], seed, idx_a)
    state_b = create_machine_state(m_b, specs[m_b], seed, idx_b)
    return state_a, state_b


class TestInitialWear:
    def test_initial_wear_nonzero(self):
        for pair in IDENTICAL_PAIRS:
            a, b = _make_pair(pair)
            a_has_wear = any(
                s.degradation_level > 0.0 for s in a.sensors.values()
            ) or a.operational_seconds > 0.0
            b_has_wear = any(
                s.degradation_level > 0.0 for s in b.sensors.values()
            ) or b.operational_seconds > 0.0
            assert a_has_wear, f"{pair[0]} should have non-zero initial wear"
            assert b_has_wear, f"{pair[1]} should have non-zero initial wear"

    def test_initial_wear_bounded(self):
        specs = build_machine_specs()
        for i, (mid, spec) in enumerate(specs.items()):
            state = create_machine_state(mid, spec, 42, i)
            for sname, sensor in state.sensors.items():
                assert sensor.degradation_level < 0.10, (
                    f"{mid}.{sname} d0={sensor.degradation_level:.4f} should be < 0.10"
                )


class TestIdenticalMachinesDiverge:
    def test_identical_machines_diverge_by_24h(self):
        for pair in IDENTICAL_PAIRS:
            a, b = _make_pair(pair)
            target_24h = 24 * 3600
            a.operational_seconds = target_24h
            b.operational_seconds = target_24h
            for sname in a.sensors:
                apply_degradation_step(a, sname, dt_seconds=1.0)
            for sname in b.sensors:
                apply_degradation_step(b, sname, dt_seconds=1.0)
            d_a = a.overall_degradation
            d_b = b.overall_degradation
            diff = abs(d_a - d_b)
            assert diff > 0.001, (
                f"{pair}: |d_a - d_b| = {diff:.6f} should be > 0.001 after 24h"
            )

    def test_identical_machines_diverge_by_72h(self):
        for pair in IDENTICAL_PAIRS:
            a, b = _make_pair(pair, seed=99)
            target_72h = 72 * 3600
            a.operational_seconds = target_72h
            b.operational_seconds = target_72h
            for sname in a.sensors:
                apply_degradation_step(a, sname, dt_seconds=1.0)
            for sname in b.sensors:
                apply_degradation_step(b, sname, dt_seconds=1.0)
            d_a = a.overall_degradation
            d_b = b.overall_degradation
            diff = abs(d_a - d_b)
            assert diff > 0.001, (
                f"{pair}: |d_a - d_b| = {diff:.6f} should be > 0.001 after 72h"
            )


class TestFailureTimeVariation:
    def test_failure_time_variation_across_seeds(self):
        specs = build_machine_specs()
        mid = "AC-201"
        idx = list(specs.keys()).index(mid)
        failure_times = []
        for seed in range(10):
            state = create_machine_state(mid, specs[mid], seed * 100, idx)
            t = state.weibull.p95_time_to_failure()
            failure_times.append(t)
        mean_ft = np.mean(failure_times)
        std_ft = np.std(failure_times)
        cv = std_ft / mean_ft if mean_ft > 0 else 0.0
        assert cv > 0.10, (
            f"Failure time CV={cv:.4f} should be > 0.10 across seeds"
        )


class TestPhysicsParamsPersisted:
    def test_physics_params_persisted(self):
        specs = build_machine_specs()
        mid = "AC-201"
        idx = list(specs.keys()).index(mid)
        state1 = create_machine_state(mid, specs[mid], 42, idx)
        state2 = create_machine_state(mid, specs[mid], 42, idx)
        for sname in state1.sensors:
            p1 = state1.sensors[sname].config.model_params
            p2 = state2.sensors[sname].config.model_params
            for key in p1:
                if isinstance(p1[key], float) and isinstance(p2.get(key), float):
                    assert p1[key] == pytest.approx(p2[key]), (
                        f"{mid}.{sname}.{key} not persisted: {p1[key]} != {p2[key]}"
                    )


class TestPerMachineJitter:
    def test_per_machine_jitter_applied(self):
        a, b = _make_pair(("AC-201", "AC-301"), seed=42)
        a.operational_seconds = 48 * 3600
        b.operational_seconds = 48 * 3600
        traj_a = []
        traj_b = []
        for _ in range(20):
            for sname in a.sensors:
                apply_degradation_step(a, sname, dt_seconds=1.0)
            for sname in b.sensors:
                apply_degradation_step(b, sname, dt_seconds=1.0)
            traj_a.append(a.overall_degradation)
            traj_b.append(b.overall_degradation)
        identical = all(
            abs(x - y) < 1e-12 for x, y in zip(traj_a, traj_b)
        )
        assert not identical, (
            "Identical-type machines should produce different trajectories"
        )
