import random
from datetime import datetime, timedelta

import polars as pl
import pytest


@pytest.fixture(scope="module")
def trained_detector():
    from src.ml.anomaly_detector import AnomalyDetector
    det = AnomalyDetector.__new__(AnomalyDetector)
    det.machine_id = "AC-201"
    det.model = None
    det.explainer = None
    det.feature_names = []
    det._contamination = 0.05
    det._model_path = "/tmp/shap_test_AC-201_anomaly.joblib"
    now = datetime.utcnow()
    rows = []
    sensors = {
        "vibration_rms": (2.5, 0.1),
        "bearing_temp": (62.0, 1.5),
        "oil_pressure": (4.5, 0.1),
        "motor_current": (21.0, 0.5),
        "outlet_pressure": (8.2, 0.1),
    }
    for i in range(300):
        for s, (mu, sigma) in sensors.items():
            rows.append({
                "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
                "sensor_name": s,
                "value": random.gauss(mu, sigma),
                "machine_phase": "HEALTHY",
                "upstream_effect": False,
            })
    df = pl.DataFrame(rows)
    det.train(df)
    return det


def test_shap_returns_top_contributing_sensor_not_root_cause(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for f in trained_detector.feature_names:
        if "vibration_rms_value" in f:
            features[f] = 15.0
        if "vibration_rms_z_score" in f:
            features[f] = 5.0
    result = trained_detector.predict(features)
    assert "top_contributing_sensor" in result
    assert "root_cause" not in result


def test_shap_values_dict_when_anomaly(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for f in trained_detector.feature_names:
        if "vibration_rms" in f:
            features[f] = 10.0
    result = trained_detector.predict(features)
    if result["is_anomaly"]:
        assert isinstance(result["shap_values"], dict)
        assert len(result["shap_values"]) > 0


def test_shap_top_sensor_is_string_or_none(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    result = trained_detector.predict(features)
    ts = result["top_contributing_sensor"]
    assert ts is None or isinstance(ts, str)


def test_shap_top_sensor_in_sensor_list(trained_detector):
    from src.data_generator.machines import MACHINE_CONFIGS
    valid_sensors = set(MACHINE_CONFIGS["AC-201"]["sensors"].keys())
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for f in trained_detector.feature_names:
        if "vibration_rms" in f:
            features[f] = 9.0
    result = trained_detector.predict(features)
    ts = result["top_contributing_sensor"]
    if ts is not None:
        assert ts in valid_sensors, f"'{ts}' not in sensor list"


def test_shap_mhi_report_uses_correct_label():
    import pathlib
    import re
    src_dir = pathlib.Path(__file__).parent / "../../src/ml"
    violations = []
    for py_file in src_dir.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        if re.search(r'["\']root_cause["\']', source):
            violations.append(str(py_file))
    assert len(violations) == 0, "'root_cause' used as key in:\n" + "\n".join(violations)
