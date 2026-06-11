"""Unit tests for src.ml.evaluation.optuna_search module."""

import json
import os

import numpy as np
import pytest

from src.ml.evaluation.optuna_search import (
    SEARCH_SPACES,
    OptunaSearchResult,
    _gb_params,
    _if_params,
    _jsonable,
    _lgbm_params,
    _make_artifact_paths,
    _make_objective,
    _xgb_params,
    tune_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_rul_dataset():
    """Small synthetic RUL dataset with 2 machines for purged CV."""
    rng = np.random.default_rng(42)
    n = 200
    ts0 = np.datetime64("2026-01-01T00:00:00", "ns")
    timestamps = ts0 + np.arange(n, dtype=np.int64) * np.timedelta64(10, "s")
    machine_ids = np.array([f"M-{i // 100:02d}" for i in range(n)], dtype=object)
    X = rng.normal(0, 1, size=(n, 4))
    y = X @ np.array([1.0, 0.5, -0.3, 0.2]) + rng.normal(0, 0.1, size=n)
    return X, y, timestamps, machine_ids


@pytest.fixture
def small_anomaly_dataset():
    """Small synthetic anomaly dataset with 2 machines."""
    rng = np.random.default_rng(42)
    n = 200
    ts0 = np.datetime64("2026-01-01T00:00:00", "ns")
    timestamps = ts0 + np.arange(n, dtype=np.int64) * np.timedelta64(10, "s")
    machine_ids = np.array([f"M-{i // 100:02d}" for i in range(n)], dtype=object)
    X = rng.normal(0, 1, size=(n, 4))
    # Binary labels: anomalous if first feature > 1.0
    y = (X[:, 0] > 1.0).astype(np.int64)
    return X, y, timestamps, machine_ids


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    """Temporary directory for Optuna artifacts."""
    d = tmp_path / "artifacts"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------


class TestSearchSpaces:
    """Test that all search spaces are registered and produce valid dicts."""

    def test_gradient_boosting_registered(self):
        assert "GradientBoosting" in SEARCH_SPACES

    def test_xgboost_registered(self):
        assert "XGBoost" in SEARCH_SPACES

    def test_isolation_forest_registered(self):
        assert "IsolationForest" in SEARCH_SPACES

    def test_search_spaces_are_callable(self):
        for name, fn in SEARCH_SPACES.items():
            assert callable(fn), f"{name} search space is not callable"

    def test_gb_params_keys(self):
        """GradientBoosting params should contain expected keys."""
        # Use a mock trial to test the param function
        class MockTrial:
            def suggest_int(self, name, low, high, step=None):
                return low
            def suggest_float(self, name, low, high, log=False):
                return low
            def suggest_categorical(self, name, choices):
                return choices[0]

        params = _gb_params(MockTrial())
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "subsample" in params
        assert "min_samples_leaf" in params

    def test_xgb_params_keys(self):
        class MockTrial:
            def suggest_int(self, name, low, high, step=None):
                return low
            def suggest_float(self, name, low, high, log=False):
                return low
            def suggest_categorical(self, name, choices):
                return choices[0]

        params = _xgb_params(MockTrial())
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "colsample_bytree" in params
        assert "reg_alpha" in params
        assert "reg_lambda" in params

    def test_if_params_keys(self):
        class MockTrial:
            def suggest_int(self, name, low, high, step=None):
                return low
            def suggest_float(self, name, low, high, log=False):
                return low
            def suggest_categorical(self, name, choices):
                return choices[0]

        params = _if_params(MockTrial())
        assert "n_estimators" in params
        assert "max_samples" in params
        assert "contamination" in params
        assert "max_features" in params

    def test_lgbm_params_keys(self):
        class MockTrial:
            def suggest_int(self, name, low, high, step=None, log=False):
                return low
            def suggest_float(self, name, low, high, log=False):
                return low
            def suggest_categorical(self, name, choices):
                return choices[0]

        params = _lgbm_params(MockTrial())
        assert "n_estimators" in params
        assert "num_leaves" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "verbosity" in params


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


class TestMakeObjective:
    """Test objective function creation and validation."""

    def test_invalid_family_raises(self, small_rul_dataset):
        X, y, timestamps, machine_ids = small_rul_dataset
        from src.ml.evaluation.temporal_split import PurgedTimeSeriesCV

        with pytest.raises(ValueError, match="family must be"):
            _make_objective(
                model_name="XGBoost",
                family="invalid_family",
                X=X,
                y=y,
                timestamps=timestamps,
                machine_ids=machine_ids,
                splitter_factory=lambda: PurgedTimeSeriesCV(n_splits=2),
                random_state=42,
                n_folds=2,
                metric_name="-mae",
            )

    def test_unknown_model_raises(self, small_rul_dataset):
        X, y, timestamps, machine_ids = small_rul_dataset
        from src.ml.evaluation.temporal_split import PurgedTimeSeriesCV

        with pytest.raises(ValueError, match="Unknown model"):
            _make_objective(
                model_name="NonExistentModel",
                family="rul",
                X=X,
                y=y,
                timestamps=timestamps,
                machine_ids=machine_ids,
                splitter_factory=lambda: PurgedTimeSeriesCV(n_splits=2),
                random_state=42,
                n_folds=2,
                metric_name="-mae",
            )

    def test_model_not_in_family_raises(self, small_rul_dataset):
        """IsolationForest has a search space but no factory for family='rul'."""
        X, y, timestamps, machine_ids = small_rul_dataset
        from src.ml.evaluation.temporal_split import PurgedTimeSeriesCV

        with pytest.raises(ValueError, match="no factory for family"):
            _make_objective(
                model_name="IsolationForest",
                family="rul",
                X=X,
                y=y,
                timestamps=timestamps,
                machine_ids=machine_ids,
                splitter_factory=lambda: PurgedTimeSeriesCV(n_splits=2),
                random_state=42,
                n_folds=2,
                metric_name="-mae",
            )

    def test_objective_returns_callable(self, small_rul_dataset):
        X, y, timestamps, machine_ids = small_rul_dataset
        from src.ml.evaluation.temporal_split import PurgedTimeSeriesCV

        objective = _make_objective(
            model_name="GradientBoosting",
            family="rul",
            X=X,
            y=y,
            timestamps=timestamps,
            machine_ids=machine_ids,
            splitter_factory=lambda: PurgedTimeSeriesCV(n_splits=2),
            random_state=42,
            n_folds=2,
            metric_name="-mae",
        )
        assert callable(objective)


# ---------------------------------------------------------------------------
# tune_model integration
# ---------------------------------------------------------------------------


class TestTuneModel:
    """Integration tests for tune_model with small datasets."""

    def test_tune_gradient_boosting_rul(self, small_rul_dataset, tmp_artifact_dir):
        X, y, timestamps, machine_ids = small_rul_dataset
        result = tune_model(
            "GradientBoosting",
            X, y, timestamps,
            family="rul",
            machine_ids=machine_ids,
            n_splits=2,
            embargo_seconds=0.0,  # No embargo for small dataset
            n_trials=3,
            random_state=42,
            artifact_dir=tmp_artifact_dir,
        )
        assert isinstance(result, OptunaSearchResult)
        assert result.model_name == "GradientBoosting"
        assert result.family == "rul"
        assert result.metric == "-mae"
        assert result.n_trials == 3
        assert isinstance(result.best_value, float)
        assert isinstance(result.best_params, dict)
        assert result.best_params_path is not None
        assert os.path.exists(result.best_params_path)

    def test_tune_xgboost_rul(self, small_rul_dataset, tmp_artifact_dir):
        X, y, timestamps, machine_ids = small_rul_dataset
        result = tune_model(
            "XGBoost",
            X, y, timestamps,
            family="rul",
            machine_ids=machine_ids,
            n_splits=2,
            embargo_seconds=0.0,
            n_trials=3,
            random_state=42,
            artifact_dir=tmp_artifact_dir,
        )
        assert result.model_name == "XGBoost"
        assert result.family == "rul"
        assert result.best_value < 0  # -mae is negative

    def test_tune_isolation_forest_anomaly(self, small_anomaly_dataset, tmp_artifact_dir):
        X, y, timestamps, machine_ids = small_anomaly_dataset
        result = tune_model(
            "IsolationForest",
            X, y, timestamps,
            family="anomaly",
            machine_ids=machine_ids,
            n_splits=2,
            embargo_seconds=0.0,
            n_trials=3,
            random_state=42,
            artifact_dir=tmp_artifact_dir,
        )
        assert result.model_name == "IsolationForest"
        assert result.family == "anomaly"
        assert result.metric == "pr_auc"

    def test_unknown_model_raises(self, small_rul_dataset, tmp_artifact_dir):
        X, y, timestamps, machine_ids = small_rul_dataset
        with pytest.raises(ValueError, match="Unknown model"):
            tune_model(
                "NonExistentModel",
                X, y, timestamps,
                family="rul",
                machine_ids=machine_ids,
                n_trials=3,
                artifact_dir=tmp_artifact_dir,
            )

    def test_invalid_family_raises(self, small_rul_dataset, tmp_artifact_dir):
        X, y, timestamps, machine_ids = small_rul_dataset
        with pytest.raises(ValueError, match="family must be"):
            tune_model(
                "XGBoost",
                X, y, timestamps,
                family="classification",
                machine_ids=machine_ids,
                n_trials=3,
                artifact_dir=tmp_artifact_dir,
            )

    def test_seed_alias(self, small_rul_dataset, tmp_artifact_dir):
        """The deprecated 'seed' parameter should work as alias for random_state."""
        X, y, timestamps, machine_ids = small_rul_dataset
        result = tune_model(
            "GradientBoosting",
            X, y, timestamps,
            family="rul",
            machine_ids=machine_ids,
            n_splits=2,
            embargo_seconds=0.0,
            n_trials=3,
            seed=123,
            artifact_dir=tmp_artifact_dir,
        )
        assert result.extra["random_state"] == 123

    def test_best_params_json_persisted(self, small_rul_dataset, tmp_artifact_dir):
        X, y, timestamps, machine_ids = small_rul_dataset
        result = tune_model(
            "GradientBoosting",
            X, y, timestamps,
            family="rul",
            machine_ids=machine_ids,
            n_splits=2,
            embargo_seconds=0.0,
            n_trials=3,
            artifact_dir=tmp_artifact_dir,
        )
        # Verify the JSON file is valid and contains expected keys
        with open(result.best_params_path) as f:
            data = json.load(f)
        assert data["model_name"] == "GradientBoosting"
        assert data["family"] == "rul"
        assert "best_params" in data
        assert "best_value" in data
        assert "tuned_at" in data


# ---------------------------------------------------------------------------
# OptunaSearchResult serialization
# ---------------------------------------------------------------------------


class TestOptunaSearchResult:
    """Test the result dataclass and its serialization."""

    def test_to_dict_basic(self):
        result = OptunaSearchResult(
            model_name="XGBoost",
            family="rul",
            best_params={"n_estimators": 100, "max_depth": 5},
            best_value=-0.5,
            n_trials=10,
            metric="-mae",
        )
        d = result.to_dict()
        assert d["model_name"] == "XGBoost"
        assert d["family"] == "rul"
        assert d["metric"] == "-mae"
        assert d["best_value"] == -0.5
        assert d["n_trials"] == 10
        assert d["best_params"] == {"n_estimators": 100, "max_depth": 5}

    def test_to_dict_with_paths(self):
        result = OptunaSearchResult(
            model_name="XGBoost",
            family="rul",
            best_params={"n_estimators": 100},
            best_value=-0.5,
            n_trials=10,
            metric="-mae",
            best_params_path="/tmp/best_params.json",
            study_html_path="/tmp/study.html",
        )
        d = result.to_dict()
        assert d["best_params_path"] == "/tmp/best_params.json"
        assert d["study_html_path"] == "/tmp/study.html"

    def test_to_dict_without_paths(self):
        result = OptunaSearchResult(
            model_name="XGBoost",
            family="rul",
            best_params={},
            best_value=0.0,
            n_trials=1,
            metric="-mae",
        )
        d = result.to_dict()
        assert "best_params_path" not in d
        assert "study_html_path" not in d

    def test_to_dict_json_serializable(self):
        result = OptunaSearchResult(
            model_name="XGBoost",
            family="rul",
            best_params={"n_estimators": 100},
            best_value=-0.5,
            n_trials=10,
            metric="-mae",
        )
        # Should not raise
        json_str = json.dumps(result.to_dict())
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test utility functions."""

    def test_jsonable_int(self):
        assert _jsonable(np.int64(42)) == 42
        assert isinstance(_jsonable(np.int64(42)), int)

    def test_jsonable_float(self):
        assert _jsonable(np.float64(3.14)) == 3.14
        assert isinstance(_jsonable(np.float64(3.14)), float)

    def test_jsonable_ndarray(self):
        arr = np.array([1, 2, 3])
        result = _jsonable(arr)
        assert result == [1, 2, 3]

    def test_jsonable_passthrough(self):
        assert _jsonable("hello") == "hello"
        assert _jsonable(42) == 42
        assert _jsonable(3.14) == 3.14
        assert _jsonable(True) is True
        assert _jsonable(None) is None

    def test_make_artifact_paths_default(self):
        paths = _make_artifact_paths(None, "XGBoost")
        assert "artifact_dir" in paths
        assert "best_params_path" in paths
        assert "study_html_path" in paths
        assert "study_pickle_path" in paths
        assert paths["artifact_dir"].endswith("XGBoost_tuning")

    def test_make_artifact_paths_custom(self, tmp_artifact_dir):
        paths = _make_artifact_paths(tmp_artifact_dir, "XGBoost")
        assert paths["artifact_dir"] == tmp_artifact_dir
        assert paths["best_params_path"].endswith("best_params.json")
        assert paths["study_html_path"].endswith("study.html")
        assert paths["study_pickle_path"].endswith("study.pkl")
