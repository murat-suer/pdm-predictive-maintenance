"""Injected faults must be demo-visible at any simulation speed.

The smoothstep boost used to be proportional to the machine's Weibull
degradation alone — on a young machine at SIMULATION_SPEED=10 that made
fault injection a silent no-op for hours. The floor + real-time
progression keep an injected fault on a minutes-scale real clock.
"""
from src.data_generator.independent_scheduler import (
    INJECTION_FLOOR_D,
    INJECTION_PROGRESSION_PER_REAL_S,
    SimulationEngine,
)
from src.data_generator.machines import ANOMALY_SCENARIOS, MACHINE_CONFIGS


def make_engine() -> SimulationEngine:
    return SimulationEngine(
        machine_specs={mid: cfg["sensors"] for mid, cfg in MACHINE_CONFIGS.items()},
        speed_multiplier=10.0,
        global_seed=42,
        anomaly_scenarios=ANOMALY_SCENARIOS,
    )


class TestInjectionPacing:
    def test_constants_reach_anomaly_zone_within_minutes(self):
        """A strength-0.5 sensor must cross the ANOMALY threshold (0.60)
        within 10 real minutes from floor + progression alone."""
        strength = 0.5
        boost_at_10min = (
            INJECTION_FLOOR_D * strength
            + strength * INJECTION_PROGRESSION_PER_REAL_S * 600.0
        )
        assert boost_at_10min >= 0.60

    def test_injection_arms_target_sensors(self):
        """fouling_spike must arm the HX-native sensors with a ramp window."""
        engine = make_engine()
        engine.inject_anomaly("HX-302", "fouling_spike", ramp_seconds=10.0)
        state = engine.states["HX-302"]
        armed = [
            s.name
            for s in state.sensors.values()
            if s.anomaly_scenario == "fouling_spike" and s.anomaly_strength > 0
        ]
        assert "pressure_drop" in armed
        assert "flow_rate" in armed

    def test_belt_slip_arms_conveyor_native_sensors(self):
        """belt_slip must move the conveyor's own sensors, not only the
        compressor-style vibration/current pair."""
        engine = make_engine()
        engine.inject_anomaly("CM-303", "belt_slip", ramp_seconds=10.0)
        state = engine.states["CM-303"]
        armed = {
            s.name
            for s in state.sensors.values()
            if s.anomaly_scenario == "belt_slip" and s.anomaly_strength > 0
        }
        assert "belt_tension" in armed
        assert "motor_load" in armed


class TestInjectionPhaseVisibility:
    def test_injected_degradation_not_clipped_by_age_ceiling(self):
        """The Weibull-age ceiling must not clip an injected fault: a young
        machine with screaming sensors stayed phase=HEALTHY, which silently
        suppressed the whole alarm chain."""
        engine = make_engine()
        engine.inject_anomaly("CM-303", "belt_slip", ramp_seconds=10.0)
        state = engine.states["CM-303"]
        # Simulate the injected boost having driven the armed sensors deep.
        for sensor in state.sensors.values():
            if sensor.anomaly_scenario == "belt_slip":
                sensor.degradation_level = 0.94
        assert state.overall_degradation > 0.50, (
            "machine-level degradation must reflect the injected fault"
        )
        state.update_phase()
        assert state.phase in ("DEGRADING", "ANOMALY")

    def test_maintenance_reset_clears_injection(self):
        """Repair fixes the injected fault — the armed scenario must not
        re-degrade the machine straight out of maintenance."""
        engine = make_engine()
        engine.inject_anomaly("CM-303", "belt_slip", ramp_seconds=10.0)
        state = engine.states["CM-303"]
        state.reset_after_maintenance()
        assert all(s.anomaly_scenario is None for s in state.sensors.values())
        assert all(s.anomaly_strength == 0.0 for s in state.sensors.values())
