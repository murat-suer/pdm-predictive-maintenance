import polars as pl


def make_test_df(machine_id="AC-201", n_rows=50):
    import random
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    rows = []
    sensors = ["vibration_rms", "bearing_temp", "oil_pressure", "motor_current", "outlet_pressure"]
    for i in range(n_rows):
        for sensor in sensors:
            rows.append({
                "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
                "sensor_name": sensor,
                "value": random.gauss(2.5, 0.15) if sensor == "vibration_rms" else random.gauss(62.0, 2.0),
                "machine_phase": "HEALTHY",
                "upstream_effect": False,
            })
    return pl.DataFrame(rows)


def test_compute_features_returns_polars():
    from src.ml.feature_engineering import compute_features
    df = make_test_df()
    result = compute_features(df)
    assert isinstance(result, pl.DataFrame)


def test_feature_columns_include_all_suffixes():
    from src.ml.feature_engineering import compute_features
    df = make_test_df(n_rows=35)
    result = compute_features(df)
    suffixes = ["_value", "_rolling_mean_5m", "_rolling_std_5m", "_rate_of_change", "_z_score", "_shift_adj_z"]
    for sensor in ["vibration_rms", "bearing_temp"]:
        for suffix in suffixes:
            col = f"{sensor}{suffix}"
            assert col in result.columns, f"Missing column: {col}"


def test_rolling_mean_converges():
    import polars as pl

    from src.ml.feature_engineering import compute_features
    n = 60
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    rows = [
        {
            "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
            "sensor_name": "vibration_rms",
            "value": 2.5,
            "machine_phase": "HEALTHY",
            "upstream_effect": False,
        }
        for i in range(n)
    ]
    df = pl.DataFrame(rows)
    result = compute_features(df)
    last = result.row(-1, named=True)
    assert abs(last["vibration_rms_rolling_mean_5m"] - 2.5) < 0.01


def test_rate_of_change_zero_on_constant_signal():
    from datetime import datetime, timedelta

    from src.ml.feature_engineering import compute_features
    now = datetime.utcnow()
    rows = [
        {
            "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
            "sensor_name": "vibration_rms",
            "value": 2.5,
            "machine_phase": "HEALTHY",
            "upstream_effect": False,
        }
        for i in range(35)
    ]
    df = pl.DataFrame(rows)
    result = compute_features(df)
    roc_values = result["vibration_rms_rate_of_change"].drop_nulls().to_list()
    assert all(abs(v) < 0.001 for v in roc_values[1:])


def test_z_score_of_nominal_near_zero():
    from datetime import datetime, timedelta

    from src.ml.feature_engineering import compute_features
    now = datetime.utcnow()
    rows = [
        {
            "timestamp": (now + timedelta(seconds=i * 10)).isoformat(),
            "sensor_name": "vibration_rms",
            "value": 2.5,
            "machine_phase": "HEALTHY",
            "upstream_effect": False,
        }
        for i in range(35)
    ]
    df = pl.DataFrame(rows)
    result = compute_features(df)
    last = result.row(-1, named=True)
    assert abs(last["vibration_rms_z_score"]) < 0.01


def test_filter_calibration_excludes_anomaly_phase():
    from src.ml.feature_engineering import filter_calibration_data
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "sensor_name": "s1",
            "value": 1.0,
            "machine_phase": "HEALTHY",
            "upstream_effect": False,
        },
        {
            "timestamp": "2026-01-01T00:00:10",
            "sensor_name": "s1",
            "value": 2.0,
            "machine_phase": "ANOMALY",
            "upstream_effect": False,
        },
        {
            "timestamp": "2026-01-01T00:00:20",
            "sensor_name": "s1",
            "value": 1.5,
            "machine_phase": "DEGRADING",
            "upstream_effect": False,
        },
    ]
    df = pl.DataFrame(rows)
    filtered = filter_calibration_data(df)
    phases = filtered["machine_phase"].to_list()
    assert "ANOMALY" not in phases


def test_filter_calibration_excludes_upstream_effect():
    from src.ml.feature_engineering import filter_calibration_data
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "sensor_name": "s1",
            "value": 1.0,
            "machine_phase": "HEALTHY",
            "upstream_effect": False,
        },
        {
            "timestamp": "2026-01-01T00:00:10",
            "sensor_name": "s1",
            "value": 150.0,
            "machine_phase": "HEALTHY",
            "upstream_effect": True,
        },
    ]
    df = pl.DataFrame(rows)
    filtered = filter_calibration_data(df)
    upstream = filtered["upstream_effect"].to_list()
    assert True not in upstream


def test_no_pandas_import_in_feature_engineering():
    import ast
    import os
    path = os.path.join(os.path.dirname(__file__), "../../src/ml/feature_engineering.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [node.module or ""]
            for name in names:
                assert "pandas" not in name, "pandas import found in feature_engineering.py"
