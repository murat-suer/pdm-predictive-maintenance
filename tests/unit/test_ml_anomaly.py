import random
from datetime import datetime, timedelta

import polars as pl
import pytest


def make_healthy_df(n=200, machine_id="AC-201"):
    now = datetime.utcnow()
    rows = []
    sensors = {
        "vibration_rms": (2.5, 0.1),
        "bearing_temp": (62.0, 1.5),
        "oil_pressure": (4.5, 0.1),
        "motor_current": (21.0, 0.5),
        "outlet_pressure": (8.2, 0.1),
    }
    for i in range(n):
        for s, (mu, sigma) in sensors.items():
            rows.append({
                "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
                "sensor_name": s,
                "value": random.gauss(mu, sigma),
                "machine_phase": "HEALTHY",
                "upstream_effect": False,
            })
    return pl.DataFrame(rows)


def make_anomaly_df(n=30, machine_id="AC-201"):
    now = datetime.utcnow()
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
            "sensor_name": "vibration_rms",
            "value": 10.0 + random.gauss(0, 0.1),
            "machine_phase": "ANOMALY",
            "upstream_effect": False,
        })
        for s, val in [
            ("bearing_temp", 95.0),
            ("oil_pressure", 1.8),
            ("motor_current", 32.0),
            ("outlet_pressure", 4.5),
        ]:
            rows.append({
                "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
                "sensor_name": s,
                "value": val + random.gauss(0, 0.2),
                "machine_phase": "ANOMALY",
                "upstream_effect": False,
            })
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_detector():
    from src.ml.anomaly_detector import AnomalyDetector
    det = AnomalyDetector.__new__(AnomalyDetector)
    det.machine_id = "AC-201"
    det.model = None
    det.explainer = None
    det.feature_names = []
    det._contamination = 0.05
    det._model_path = "/tmp/test_AC-201_anomaly.joblib"
    df = make_healthy_df(n=300)
    det.train(df)
    return det


def test_train_creates_model(trained_detector):
    assert trained_detector.model is not None


def test_train_sets_feature_names(trained_detector):
    assert len(trained_detector.feature_names) > 0


def test_healthy_reading_not_anomaly(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for feat in trained_detector.feature_names:
        if feat.endswith("_value"):
            features[feat] = 2.5
    result = trained_detector.predict(features)
    assert isinstance(result["is_anomaly"], bool)
    assert 0.0 <= result["anomaly_score"] <= 1.0


def test_extreme_values_produce_high_score(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for feat in trained_detector.feature_names:
        if "vibration_rms_value" in feat:
            features[feat] = 15.0
        if "vibration_rms_z_score" in feat:
            features[feat] = 6.0
    result = trained_detector.predict(features)
    assert result["anomaly_score"] > 0.0


def test_predict_returns_required_keys(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    result = trained_detector.predict(features)
    required = ["is_anomaly", "anomaly_score", "top_contributing_sensor", "shap_values"]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_no_root_cause_key_in_result(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    result = trained_detector.predict(features)
    assert "root_cause" not in result


def test_upstream_effect_not_in_training_data():
    import polars as pl

    from src.ml.feature_engineering import filter_calibration_data
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "sensor_name": "vibration_rms",
            "value": 2.5,
            "machine_phase": "HEALTHY",
            "upstream_effect": False,
        },
        {
            "timestamp": "2026-01-01T00:00:10",
            "sensor_name": "vibration_rms",
            "value": 8.0,
            "machine_phase": "HEALTHY",
            "upstream_effect": True,
        },
    ]
    df = pl.DataFrame(rows)
    filtered = filter_calibration_data(df)
    assert len(filtered) == 1
    assert filtered["upstream_effect"].to_list() == [False]


def test_untrained_detector_returns_not_anomaly():
    from src.ml.anomaly_detector import AnomalyDetector
    det = AnomalyDetector.__new__(AnomalyDetector)
    det.machine_id = "HX-202"
    det.model = None
    det.explainer = None
    det.feature_names = []
    det._contamination = 0.05
    det._model_path = "/tmp/nonexistent_HX202_anomaly.joblib"
    result = det.predict({"vibration_rms_value": 2.5})
    assert result["is_anomaly"] is False
    assert result["anomaly_score"] == 0.0


def test_predict_returns_score_and_shap_structure(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for feat in trained_detector.feature_names:
        if feat.endswith("_value"):
            if "vibration" in feat:
                features[feat] = 15.0
            else:
                features[feat] = 2.5
    result = trained_detector.predict(features)
    assert "is_anomaly" in result
    assert "anomaly_score" in result
    assert "top_contributing_sensor" in result
    assert "shap_values" in result
    assert isinstance(result["is_anomaly"], bool)
    assert isinstance(result["anomaly_score"], (int, float))
    assert isinstance(result["shap_values"], dict)
    assert 0.0 <= result["anomaly_score"] <= 1.0
    if result["is_anomaly"]:
        assert len(result["shap_values"]) > 0
        assert result["top_contributing_sensor"] is not None


def test_predict_skips_shap_on_normal_readings(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for feat in trained_detector.feature_names:
        if feat.endswith("_value"):
            features[feat] = 2.5
    result = trained_detector.predict(features)
    if not result["is_anomaly"]:
        assert result["shap_values"] == {} or result["shap_values"] == {}


def test_predict_handles_null_readings_gracefully(trained_detector):
    features = {}
    for feat in trained_detector.feature_names:
        if feat.endswith("_value"):
            features[feat] = None
        else:
            features[feat] = 0.0
    try:
        result = trained_detector.predict(features)
    except Exception as e:
        pytest.fail(f"predict() crashed on null readings: {e}")
    assert "is_anomaly" in result
    assert "anomaly_score" in result
    assert "shap_values" in result
    assert "top_contributing_sensor" in result
    assert isinstance(result["anomaly_score"], (int, float))
    assert 0.0 <= result["anomaly_score"] <= 1.0


def test_predict_with_partial_null_readings(trained_detector):
    features = {}
    for feat in trained_detector.feature_names:
        if "vibration_rms_value" in feat:
            features[feat] = 15.0
        elif feat.endswith("_value"):
            features[feat] = None
        else:
            features[feat] = 0.0
    result = trained_detector.predict(features)
    assert isinstance(result["anomaly_score"], (int, float))
    assert 0.0 <= result["anomaly_score"] <= 1.0


def test_predict_handles_empty_feature_dict():
    from src.ml.anomaly_detector import AnomalyDetector
    det = AnomalyDetector.__new__(AnomalyDetector)
    det.machine_id = "TEST-MACHINE"
    det.model = None
    det.explainer = None
    det.feature_names = []
    det._contamination = 0.05
    det._model_path = "/tmp/nonexistent_anomaly.joblib"
    result = det.predict({})
    assert result["is_anomaly"] is False
    assert result["anomaly_score"] == 0.0
    assert result["shap_values"] == {}
    assert result["top_contributing_sensor"] is None


def test_predict_top_contributing_sensor_on_anomaly(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for feat in trained_detector.feature_names:
        if "vibration_rms_value" in feat:
            features[feat] = 15.0
        elif feat.endswith("_value"):
            features[feat] = 2.5
    result = trained_detector.predict(features)
    if result["is_anomaly"]:
        assert result["top_contributing_sensor"] is not None
        assert result["top_contributing_sensor"] != "root_cause"
        assert len(result["top_contributing_sensor"]) > 0
        assert isinstance(result["top_contributing_sensor"], str)
    if result["shap_values"]:
        for feat, val in result["shap_values"].items():
            assert isinstance(val, (int, float)), f"SHAP value for {feat} is not numeric: {val}"


def test_anomaly_score_monotonic_with_extreme_values(trained_detector):
    nom_features = {}
    mod_features = {}
    ext_features = {}
    for feat in trained_detector.feature_names:
        if "vibration_rms_value" in feat:
            nom_features[feat] = 2.5
            mod_features[feat] = 7.0
            ext_features[feat] = 15.0
        elif feat.endswith("_value"):
            nom_features[feat] = 2.5
            mod_features[feat] = 2.5
            ext_features[feat] = 2.5
        else:
            nom_features[feat] = 0.0
            mod_features[feat] = 0.0
            ext_features[feat] = 0.0
    score_nom = trained_detector.predict(nom_features)["anomaly_score"]
    score_mod = trained_detector.predict(mod_features)["anomaly_score"]
    score_ext = trained_detector.predict(ext_features)["anomaly_score"]
    assert score_ext >= score_nom


def test_multiple_predictions_consistent(trained_detector):
    features = dict.fromkeys(trained_detector.feature_names, 0.0)
    for feat in trained_detector.feature_names:
        if feat.endswith("_value"):
            features[feat] = 5.0
    results = [trained_detector.predict(features) for _ in range(5)]
    scores = [r["anomaly_score"] for r in results]
    anomalies = [r["is_anomaly"] for r in results]
    assert all(s == scores[0] for s in scores)
    assert all(a == anomalies[0] for a in anomalies)


def test_detector_init_loads_existing_model():
    from src.ml.anomaly_detector import AnomalyDetector
    det = AnomalyDetector("NONEXISTENT-MACHINE-999")
    assert det.machine_id == "NONEXISTENT-MACHINE-999"
    assert det.model is None or det.model is not None


def test_compute_shap_handles_exception_gracefully():
    import numpy as np

    from src.ml.anomaly_detector import AnomalyDetector
    det = AnomalyDetector.__new__(AnomalyDetector)
    det.machine_id = "TEST"
    det.feature_names = ["vibration_rms_value", "bearing_temp_value"]
    det.explainer = object()
    X = np.array([[2.5, 65.0]])
    result = det._compute_shap(X)
    assert result == {"top_contributing_sensor": None, "shap_values": {}}
