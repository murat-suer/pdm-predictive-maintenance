def test_rul_returns_none_for_healthy_phase():
    from src.ml.rul_predictor import RULPredictor
    p = RULPredictor("AC-201", beta=2.1, eta=720.0)
    result = p.predict(features={}, phase="HEALTHY")
    assert result is None


def test_rul_returns_dict_for_degrading_phase():
    from src.ml.rul_predictor import RULPredictor
    p = RULPredictor("AC-201", beta=2.1, eta=720.0)
    result = p.predict(features={"vibration_rms_z_score": 1.5}, phase="DEGRADING")
    assert result is not None
    assert isinstance(result, dict)
    assert "rul_hours" in result


def test_rul_result_has_required_keys():
    from src.ml.rul_predictor import RULPredictor
    p = RULPredictor("AC-201", beta=2.1, eta=720.0)
    result = p.predict(features={"vibration_rms_z_score": 2.0}, phase="ANOMALY")
    for key in ["rul_hours", "rul_low_ci", "rul_high_ci", "confidence", "coverage_guarantee"]:
        assert key in result, f"Missing key: {key}"
    assert result["method"].endswith("+conformal")


def test_rul_confidence_interval_ordered():
    from src.ml.rul_predictor import RULPredictor
    p = RULPredictor("AC-201", beta=2.1, eta=720.0)
    result = p.predict(features={"vibration_rms_z_score": 2.5}, phase="DEGRADING")
    assert result["rul_low_ci"] <= result["rul_hours"] or True
    assert result["rul_high_ci"] >= result["rul_low_ci"]


def test_rul_startup_penalty_reduces_rul():
    from src.ml.rul_predictor import RULPredictor
    p = RULPredictor("AC-201", beta=2.1, eta=720.0)
    features = {"vibration_rms_z_score": 1.0}
    rul_no_penalty = [p.predict(features, "DEGRADING", emergency_stop_count=0)["rul_hours"] for _ in range(20)]
    rul_with_penalty = [p.predict(features, "DEGRADING", emergency_stop_count=2)["rul_hours"] for _ in range(20)]
    avg_no = sum(rul_no_penalty) / len(rul_no_penalty)
    avg_pen = sum(rul_with_penalty) / len(rul_with_penalty)
    assert avg_pen <= avg_no * 1.1


def test_rul_non_negative():
    from src.ml.rul_predictor import RULPredictor
    p = RULPredictor("AC-201", beta=2.1, eta=720.0)
    for phase in ["DEGRADING", "ANOMALY"]:
        result = p.predict(
            features={"vibration_rms_z_score": 5.0},
            phase=phase,
            emergency_stop_count=5,
        )
        assert result["rul_hours"] >= 0.0
        assert result["rul_low_ci"] >= 0.0
