def make_mock_detector(is_anomaly_return=True):
    class MockDetector:
        machine_id = "AC-201"
        feature_names = ["vibration_rms_value", "vibration_rms_z_score"]

        def predict(self, features):
            return {
                "is_anomaly": is_anomaly_return,
                "anomaly_score": 0.85 if is_anomaly_return else 0.1,
                "top_contributing_sensor": "vibration_rms",
                "shap_values": {"vibration_rms_value": 0.5},
            }

    return MockDetector()


def test_canary_probe_returns_log_entry():
    from src.ml.model_health import CanaryProbeSystem
    cs = CanaryProbeSystem()
    det = make_mock_detector(is_anomaly_return=True)
    result = cs.run_probe("AC-201", "AC", det, triggered_by="SCHEDULED")
    required = ["probe_id", "machine_id", "probe_type", "started_at", "detected", "expected", "success", "triggered_by"]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_canary_probe_success_when_detected():
    from src.ml.model_health import CanaryProbeSystem
    cs = CanaryProbeSystem()
    det = make_mock_detector(is_anomaly_return=True)
    result = cs.run_probe("AC-201", "AC", det)
    assert result["success"] is True
    assert result["detected"] is True
    assert result["recalibration_triggered"] is False


def test_canary_probe_failure_when_not_detected():
    from src.ml.model_health import CanaryProbeSystem
    cs = CanaryProbeSystem()
    det = make_mock_detector(is_anomaly_return=False)
    result = cs.run_probe("AC-201", "AC", det)
    assert result["success"] is False
    assert result["recalibration_triggered"] is True


def test_canary_probe_id_is_uuid():
    import re

    from src.ml.model_health import CanaryProbeSystem
    UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    cs = CanaryProbeSystem()
    det = make_mock_detector()
    result = cs.run_probe("HX-202", "HX", det)
    assert UUID_RE.match(result["probe_id"]), f"Invalid UUID: {result['probe_id']}"


def test_canary_probe_hx_scenario_selected():
    from src.ml.model_health import CanaryProbeSystem
    cs = CanaryProbeSystem()
    det = make_mock_detector()
    hx_scenarios = [s["name"] for s in cs.PROBE_SCENARIOS["HX"]]
    result = cs.run_probe("HX-202", "HX", det)
    assert result["scenario"] in hx_scenarios
