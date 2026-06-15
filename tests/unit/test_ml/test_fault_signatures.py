"""
tests/unit/test_ml/test_fault_signatures.py
============================================
Deterministic fault-signature test for all 6 production machines.

For each machine we:
1. Build mock MachineBaseline rows from the machine's nominal_mu / nominal_sigma
   (exactly what the healthy baseline looks like in the live system).
2. Compute sensor_readings that represent a mid-degradation anomaly (~50% of the
   nominal→warning range) driven by that machine's dominant degradation sensors.
3. Compute SHAP values proportional to each sensor's degradation_weight (the
   dominant sensors have the highest SHAP contribution, which is how the live
   IsolationForest-SHAP pipeline works).
4. Assert FaultClassifier.classify() returns the expected fault_type.
5. Assert same-type machine pairs produce DIFFERENT fault types.

This test is the authoritative correctness gate for the fault-signature layer.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.data_generator.machines import MACHINE_CONFIGS
from src.ml.fault_classifier import (
    BEARING_FAULT,
    BELT_SLIP,
    FLOW_RESTRICTION,
    FOULING,
    UNCLASSIFIED_ANOMALY,
    VALVE_LEAK,
    FaultClassifier,
)

# ── Target fault-type matrix ──────────────────────────────────────────────────
# Agreed target: each machine's declared failure_mode, dominant degradation
# sensors, and classifier output must all agree.
TARGET_FAULT = {
    "AC-201": BEARING_FAULT,
    "AC-301": VALVE_LEAK,
    "CM-203": BEARING_FAULT,
    "CM-303": BELT_SLIP,
    "HX-202": FOULING,
    "HX-302": FLOW_RESTRICTION,
}


def _make_classifier_with_baseline(machine_id: str) -> FaultClassifier:
    """
    Return a FaultClassifier whose _get_baseline() returns the healthy
    nominal_mu / nominal_sigma for every sensor of *machine_id*.

    This mirrors how the live system seeds MachineBaseline rows from the
    first N healthy readings (mean ≈ nominal_mu, std ≈ nominal_sigma).
    """
    cfg = MACHINE_CONFIGS[machine_id]["sensors"]

    @contextmanager
    def _mock_db_factory():
        rows = []
        for sensor_name, scfg in cfg.items():
            row = MagicMock()
            row.sensor = sensor_name
            row.mean_value = scfg["nominal_mu"]
            row.std_value = scfg["nominal_sigma"]
            rows.append(row)

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = rows
        yield mock_session

    return FaultClassifier(db_session_factory=_mock_db_factory)


def _build_anomaly_inputs(machine_id: str) -> tuple[dict[str, float], dict[str, float], str]:
    """
    Derive sensor_readings, shap_values, and top_contributing_sensor for a
    representative mid-degradation anomaly on *machine_id*.

    Sensor readings
    ---------------
    For each sensor:
        reading = nominal_mu + degradation_direction * degradation_weight * k * nominal_sigma
    where k = 4.0 (a 4-sigma push in the degradation direction).

    This ensures:
    - Dominant-weight sensors are pushed well past the 2σ rule threshold.
    - Low-weight sensors see minor drift and stay below threshold.

    SHAP values
    -----------
    Proportional to degradation_weight (IsolationForest SHAP contribution
    correlates with how much a sensor contributes to the anomaly, which in turn
    reflects how fast it degrades).
    """
    cfg = MACHINE_CONFIGS[machine_id]["sensors"]

    sensor_readings: dict[str, float] = {}
    shap_values: dict[str, float] = {}

    # k controls the per-sensor sigma push. Using weight * k means the dominant
    # sensor (weight=0.45) is pushed 9σ and the smallest (weight=0.05) only 1σ.
    # k=20 is chosen so dominant sensors (≥0.30 weight) exceed the 2σ k_sigma
    # threshold while low-weight sensors stay below it, making the pattern
    # selective. This mirrors organic degradation at ~50% lifecycle: dominant
    # sensors degrade fast and reach the detection window first.
    k = 20.0  # number of sigma to push each sensor proportionally by weight

    for sensor_name, scfg in cfg.items():
        mu = scfg["nominal_mu"]
        sigma = scfg["nominal_sigma"]
        weight = scfg["degradation_weight"]
        direction = scfg["degradation_direction"]

        # Push reading in the degradation direction, scaled by weight
        reading = mu + direction * weight * k * sigma
        sensor_readings[sensor_name] = reading

        # SHAP is positive for anomalous contribution, proportional to weight
        shap_values[sensor_name] = weight

    # Top contributing sensor is the one with highest SHAP (= highest weight)
    top_sensor = max(shap_values, key=lambda s: shap_values[s])

    return sensor_readings, shap_values, top_sensor


# ── Per-machine parametrised test ─────────────────────────────────────────────

MACHINE_PARAMS = [
    ("AC-201", BEARING_FAULT),
    ("AC-301", VALVE_LEAK),
    ("CM-203", BEARING_FAULT),
    ("CM-303", BELT_SLIP),
    ("HX-202", FOULING),
    ("HX-302", FLOW_RESTRICTION),
]


@pytest.mark.parametrize("machine_id,expected_fault", MACHINE_PARAMS)
def test_organic_anomaly_classifies_correctly(machine_id: str, expected_fault: str) -> None:
    """
    For each machine, a mid-degradation organic anomaly must classify to
    the machine's target fault type (not UNCLASSIFIED_ANOMALY).
    """
    clf = _make_classifier_with_baseline(machine_id)
    sensor_readings, shap_values, top_sensor = _build_anomaly_inputs(machine_id)
    machine_type = machine_id.split("-")[0]

    result = clf.classify(
        machine_id=machine_id,
        machine_type=machine_type,
        anomaly_score=0.75,
        shap_values=shap_values,
        sensor_readings=sensor_readings,
        top_contributing_sensor=top_sensor,
    )

    assert result.fault_type != UNCLASSIFIED_ANOMALY, (
        f"{machine_id}: expected {expected_fault} but got UNCLASSIFIED_ANOMALY "
        f"(confidence={result.fault_confidence:.3f}, "
        f"matched_rules={result.matched_rules})"
    )
    assert result.fault_type == expected_fault, (
        f"{machine_id}: expected {expected_fault} but got {result.fault_type} "
        f"(confidence={result.fault_confidence:.3f})"
    )


# ── Same-type pairs must differ ───────────────────────────────────────────────

def test_ac_pair_classify_differently() -> None:
    """AC-201 and AC-301 must produce different fault types."""
    def _classify(machine_id: str) -> str:
        clf = _make_classifier_with_baseline(machine_id)
        readings, shap, top = _build_anomaly_inputs(machine_id)
        r = clf.classify(
            machine_id=machine_id,
            machine_type="AC",
            anomaly_score=0.75,
            shap_values=shap,
            sensor_readings=readings,
            top_contributing_sensor=top,
        )
        return r.fault_type

    ft_201 = _classify("AC-201")
    ft_301 = _classify("AC-301")
    assert ft_201 != ft_301, (
        f"AC-201 and AC-301 both classified as {ft_201}; they must differ"
    )


def test_cm_pair_classify_differently() -> None:
    """CM-203 and CM-303 must produce different fault types."""
    def _classify(machine_id: str) -> str:
        clf = _make_classifier_with_baseline(machine_id)
        readings, shap, top = _build_anomaly_inputs(machine_id)
        r = clf.classify(
            machine_id=machine_id,
            machine_type="CM",
            anomaly_score=0.75,
            shap_values=shap,
            sensor_readings=readings,
            top_contributing_sensor=top,
        )
        return r.fault_type

    ft_203 = _classify("CM-203")
    ft_303 = _classify("CM-303")
    assert ft_203 != ft_303, (
        f"CM-203 and CM-303 both classified as {ft_203}; they must differ"
    )


def test_hx_pair_classify_differently() -> None:
    """HX-202 and HX-302 must produce different fault types."""
    def _classify(machine_id: str) -> str:
        clf = _make_classifier_with_baseline(machine_id)
        readings, shap, top = _build_anomaly_inputs(machine_id)
        r = clf.classify(
            machine_id=machine_id,
            machine_type="HX",
            anomaly_score=0.75,
            shap_values=shap,
            sensor_readings=readings,
            top_contributing_sensor=top,
        )
        return r.fault_type

    ft_202 = _classify("HX-202")
    ft_302 = _classify("HX-302")
    assert ft_202 != ft_302, (
        f"HX-202 and HX-302 both classified as {ft_202}; they must differ"
    )


# ── Fleet-wide uniqueness (soft goal) ─────────────────────────────────────────

def test_fleet_has_at_least_four_distinct_fault_types() -> None:
    """
    Across all 6 machines, at least 4 distinct fault types should be present.
    Recommended target matrix yields 5 distinct types:
    BEARING_FAULT, VALVE_LEAK, BELT_SLIP, FOULING, FLOW_RESTRICTION.
    """
    fault_types = set()
    for machine_id in MACHINE_CONFIGS:
        clf = _make_classifier_with_baseline(machine_id)
        readings, shap, top = _build_anomaly_inputs(machine_id)
        machine_type = machine_id.split("-")[0]
        r = clf.classify(
            machine_id=machine_id,
            machine_type=machine_type,
            anomaly_score=0.75,
            shap_values=shap,
            sensor_readings=readings,
            top_contributing_sensor=top,
        )
        if r.fault_type != UNCLASSIFIED_ANOMALY:
            fault_types.add(r.fault_type)

    assert len(fault_types) >= 4, (
        f"Expected ≥4 distinct fault types across 6 machines, got {len(fault_types)}: {fault_types}"
    )
