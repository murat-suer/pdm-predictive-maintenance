"""Unit tests for src.ml.anomaly_detector module."""

import os

import numpy as np
import polars as pl
import pytest

from src.ml.anomaly_detector import AnomalyDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_model_store(tmp_path, monkeypatch):
    """Redirect MODEL_STORE to a temp directory for test isolation."""
    store_dir = tmp_path / "model_store"
    store_dir.mkdir()
    monkeypatch.setenv("MODEL_STORE_PATH", str(store_dir))
    return store_dir


@pytest.fixture
def training_df():
    """
    Create a Polars DataFrame with enough data for AnomalyDetector.train().

    Needs:
    - At least 50 unique timestamps with NORMAL/DEGRADING phase
    - Multiple sensors with timestamp, sensor_name, value columns
    - machine_phase column for filter_calibration_data
    """
    n_timestamps = 60  # > 50 minimum
    sensors = ["vibration_rms", "bearing_temp", "pressure_drop", "flow_rate"]
    rng = np.random.RandomState(42)

    rows = []
    for i in range(n_timestamps):
        ts = f"2024-01-01T00:{i:02d}:00"
        for sensor in sensors:
            rows.append({
                "timestamp": ts,
                "sensor_name": sensor,
                "value": rng.randn() * 0.5 + (50.0 if sensor == "bearing_temp" else 10.0),
                "machine_phase": "NORMAL",
                "machine_id": "TEST-001",
            })
    return pl.DataFrame(rows)


@pytest.fixture
def small_training_df():
    """DataFrame with fewer than 50 rows — should trigger insufficient data warning."""
    rows = []
    for i in range(10):
        ts = f"2024-01-01T00:{i:02d}:00"
        rows.append({
            "timestamp": ts,
            "sensor_name": "vibration_rms",
            "value": float(i),
            "machine_phase": "NORMAL",
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestAnomalyDetectorInit:
    def test_basic_init(self, tmp_model_store):
        """Detector initializes with machine_id and no model."""
        det = AnomalyDetector("AC-201")
        assert det.machine_id == "AC-201"
        assert det.model is None
        assert det.explainer is None
        assert det.feature_names == []

    def test_custom_contamination(self, tmp_model_store):
        """Custom contamination parameter is stored."""
        det = AnomalyDetector("AC-201", contamination=0.1)
        assert det._contamination == 0.1

    def test_default_contamination(self, tmp_model_store):
        """Default contamination comes from env or constant."""
        det = AnomalyDetector("AC-201")
        assert 0.0 < det._contamination <= 1.0

    def test_model_path_set(self, tmp_model_store):
        """Model path should point to the temp store."""
        det = AnomalyDetector("AC-201")
        assert "AC-201_anomaly.joblib" in det._model_path


# ---------------------------------------------------------------------------
# Tests: predict() with no model
# ---------------------------------------------------------------------------


class TestPredictNoModel:
    def test_returns_default_dict(self, tmp_model_store):
        """predict() with no trained model returns safe defaults."""
        det = AnomalyDetector("AC-201")
        result = det.predict({"vibration_rms_value": 1.0})
        assert result["is_anomaly"] is False
        assert result["anomaly_score"] == 0.0
        assert result["top_contributing_sensor"] is None
        assert result["shap_values"] == {}


# ---------------------------------------------------------------------------
# Tests: train()
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_success(self, tmp_model_store, training_df):
        """Training with valid data produces a fitted model."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)
        assert det.model is not None
        assert det.explainer is not None
        assert len(det.feature_names) > 0

    def test_train_insufficient_data(self, tmp_model_store, small_training_df):
        """Training with < 50 rows should not produce a model."""
        det = AnomalyDetector("TEST-001")
        det.train(small_training_df)
        assert det.model is None

    def test_train_creates_model_file(self, tmp_model_store, training_df):
        """Training should persist model to disk."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)
        assert os.path.exists(det._model_path)

    def test_train_creates_sha256_file(self, tmp_model_store, training_df):
        """Training should create SHA-256 hash file."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)
        assert os.path.exists(f"{det._model_path}.sha256")

    def test_train_creates_model_card(self, tmp_model_store, training_df):
        """Training should create a model card JSON."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)
        card_path = det._model_path.replace(".joblib", ".model_card.json")
        assert os.path.exists(card_path)

    def test_train_excludes_anomaly_phases(self, tmp_model_store):
        """Rows with ANOMALY phase should be excluded from training."""
        n = 100  # enough timestamps so that after filtering, 50+ remain
        rng = np.random.RandomState(42)
        rows = []
        for i in range(n):
            ts = f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00"
            phase = "ANOMALY" if i < 20 else "NORMAL"  # 80 NORMAL timestamps remain
            for sensor in ["vibration_rms", "bearing_temp"]:
                rows.append({
                    "timestamp": ts,
                    "sensor_name": sensor,
                    "value": rng.randn() + 10.0,
                    "machine_phase": phase,
                })
        df = pl.DataFrame(rows)
        det = AnomalyDetector("TEST-001")
        det.train(df)
        # 80 NORMAL timestamps remain after filtering → > 50 → model trains
        assert det.model is not None


# ---------------------------------------------------------------------------
# Tests: predict() after training
# ---------------------------------------------------------------------------


class TestPredictAfterTraining:
    def test_predict_normal_data(self, tmp_model_store, training_df):
        """Prediction on normal data should return valid structure."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        # Build a features_row from the trained feature names
        features_row = dict.fromkeys(det.feature_names, 10.0)
        result = det.predict(features_row)

        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "top_contributing_sensor" in result
        assert "shap_values" in result
        assert isinstance(result["is_anomaly"], bool)
        assert isinstance(result["anomaly_score"], float)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_predict_anomalous_data(self, tmp_model_store, training_df):
        """Prediction on extreme outlier data should flag anomaly."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        # Create extreme outlier features
        features_row = dict.fromkeys(det.feature_names, 99999.0)
        result = det.predict(features_row)

        # With extreme outliers, IsolationForest should flag as anomaly
        assert result["is_anomaly"] is True
        assert result["anomaly_score"] > 0.0

    def test_predict_returns_shap_on_anomaly(self, tmp_model_store, training_df):
        """When anomaly detected, SHAP values should be computed."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        # Extreme outlier to trigger anomaly
        features_row = dict.fromkeys(det.feature_names, 99999.0)
        result = det.predict(features_row)

        if result["is_anomaly"]:
            assert isinstance(result["shap_values"], dict)
            assert len(result["shap_values"]) > 0
            # top_contributing_sensor should be set if there are _value features
            value_feats = [f for f in det.feature_names if f.endswith("_value")]
            if value_feats:
                assert result["top_contributing_sensor"] is not None

    def test_predict_missing_features_default_zero(self, tmp_model_store, training_df):
        """Missing features in predict dict should default to 0.0."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        # Empty features dict — all should default to 0.0
        result = det.predict({})
        assert "is_anomaly" in result
        assert isinstance(result["anomaly_score"], float)

    def test_predict_nan_features_handled(self, tmp_model_store, training_df):
        """NaN values in features should be handled (converted to 0)."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        features_row = {f: float("nan") for f in det.feature_names}
        result = det.predict(features_row)
        assert "is_anomaly" in result
        # Should not crash


# ---------------------------------------------------------------------------
# Tests: Model persistence (save/load)
# ---------------------------------------------------------------------------


class TestModelPersistence:
    def test_save_and_load(self, tmp_model_store, training_df):
        """Model saved during training can be loaded by a new detector instance."""
        det1 = AnomalyDetector("TEST-001")
        det1.train(training_df)
        original_features = det1.feature_names.copy()

        # Create new detector — should auto-load from disk
        det2 = AnomalyDetector("TEST-001")
        assert det2.model is not None
        assert det2.feature_names == original_features

    def test_loaded_model_predicts(self, tmp_model_store, training_df):
        """Loaded model should produce predictions."""
        det1 = AnomalyDetector("TEST-001")
        det1.train(training_df)

        det2 = AnomalyDetector("TEST-001")
        features_row = dict.fromkeys(det2.feature_names, 10.0)
        result = det2.predict(features_row)
        assert "is_anomaly" in result

    def test_sha256_integrity_check(self, tmp_model_store, training_df):
        """Tampered model file should raise RuntimeError on load."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        # Tamper with the model file
        with open(det._model_path, "ab") as f:
            f.write(b"TAMPERED")

        # Loading should raise RuntimeError due to hash mismatch
        with pytest.raises(RuntimeError, match="integrity check failed"):
            AnomalyDetector("TEST-001")

    def test_bootstrap_hash_generation(self, tmp_model_store, training_df):
        """If hash file is missing but model exists, bootstrap hash is generated."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        # Delete hash file
        hash_path = f"{det._model_path}.sha256"
        os.remove(hash_path)

        # Loading should regenerate the hash (no error)
        det2 = AnomalyDetector("TEST-001")
        assert os.path.exists(hash_path)

    def test_stale_artifact_version(self, tmp_model_store, training_df, monkeypatch):
        """Model with different artifact version should not load."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        # Modify the saved artifact version
        import joblib
        data = joblib.load(det._model_path)
        data["artifact_version"] = "old-version"
        joblib.dump(data, det._model_path)

        # Regenerate hash for tampered file
        import hashlib
        hasher = hashlib.sha256()
        with open(det._model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        with open(f"{det._model_path}.sha256", "w") as f:
            f.write(hasher.hexdigest())

        # Loading should succeed but model stays None (stale version)
        det2 = AnomalyDetector("TEST-001")
        assert det2.model is None


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe(self, tmp_model_store):
        """Empty DataFrame should not crash train()."""
        df = pl.DataFrame({
            "timestamp": pl.Series([], dtype=pl.Utf8),
            "sensor_name": pl.Series([], dtype=pl.Utf8),
            "value": pl.Series([], dtype=pl.Float64),
            "machine_phase": pl.Series([], dtype=pl.Utf8),
        })
        det = AnomalyDetector("TEST-001")
        det.train(df)
        assert det.model is None

    def test_all_anomaly_phase(self, tmp_model_store):
        """All rows with ANOMALY phase → no training data → model stays None."""
        rows = []
        for i in range(60):
            ts = f"2024-01-01T00:{i:02d}:00"
            rows.append({
                "timestamp": ts,
                "sensor_name": "vibration_rms",
                "value": float(i),
                "machine_phase": "ANOMALY",
            })
        df = pl.DataFrame(rows)
        det = AnomalyDetector("TEST-001")
        det.train(df)
        assert det.model is None

    def test_single_sensor(self, tmp_model_store):
        """Training with a single sensor should still work."""
        n = 60
        rng = np.random.RandomState(42)
        rows = []
        for i in range(n):
            ts = f"2024-01-01T00:{i:02d}:00"
            rows.append({
                "timestamp": ts,
                "sensor_name": "vibration_rms",
                "value": rng.randn() + 10.0,
                "machine_phase": "NORMAL",
            })
        df = pl.DataFrame(rows)
        det = AnomalyDetector("TEST-001")
        det.train(df)
        assert det.model is not None
        assert len(det.feature_names) > 0

    def test_predict_with_extra_features(self, tmp_model_store, training_df):
        """Extra features in predict dict should be ignored."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        features_row = dict.fromkeys(det.feature_names, 10.0)
        features_row["extra_unknown_feature"] = 999.0
        result = det.predict(features_row)
        assert "is_anomaly" in result

    def test_anomaly_score_bounded(self, tmp_model_store, training_df):
        """Anomaly score should always be in [0.0, 1.0]."""
        det = AnomalyDetector("TEST-001")
        det.train(training_df)

        for multiplier in [0.0, 1.0, 100.0, 99999.0, -99999.0]:
            features_row = dict.fromkeys(det.feature_names, multiplier)
            result = det.predict(features_row)
            assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_degrading_phase_included_in_training(self, tmp_model_store):
        """DEGRADING phase should be included in training data."""
        n = 60
        rng = np.random.RandomState(42)
        rows = []
        for i in range(n):
            ts = f"2024-01-01T00:{i:02d}:00"
            phase = "DEGRADING" if i >= 40 else "NORMAL"
            for sensor in ["vibration_rms", "bearing_temp"]:
                rows.append({
                    "timestamp": ts,
                    "sensor_name": sensor,
                    "value": rng.randn() + 10.0,
                    "machine_phase": phase,
                })
        df = pl.DataFrame(rows)
        det = AnomalyDetector("TEST-001")
        det.train(df)
        # DEGRADING is not in exclude_phases, so it should be included
        assert det.model is not None
