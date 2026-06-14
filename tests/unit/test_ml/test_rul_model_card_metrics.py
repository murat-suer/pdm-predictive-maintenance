"""
Tests for held-out RUL metrics written into model cards.

Covers:
- compute_rul_holdout_metrics: sufficient data → real finite MAE/RMSE
- compute_rul_holdout_metrics: insufficient data → graceful None + reason
- RULPredictor.train: model card contains real metric after training on
  sufficient data
- RULPredictor.train: model card degrades gracefully when the holdout
  function is given too few rows (cold-start path)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Redirect model store to tmp_path so tests never touch the real store."""
    monkeypatch.setenv("MODEL_STORE_PATH", str(tmp_path))
    from src.ml.model_store import paths as _paths
    monkeypatch.setattr(_paths, "get_model_store", lambda: tmp_path)


def _make_rul_data(n: int, n_features: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, y) where y has a learnable relationship to X."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Simple linear signal so XGBoost can fit meaningfully
    weights = rng.randn(n_features)
    y = np.abs(X @ weights) * 100 + rng.rand(n) * 10 + 20
    return X.astype(np.float64), y.astype(np.float64)


# ---------------------------------------------------------------------------
# Unit tests for compute_rul_holdout_metrics
# ---------------------------------------------------------------------------

class TestComputeRULHoldoutMetrics:
    def test_sufficient_data_returns_measured_status(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(200)
        result = compute_rul_holdout_metrics(X, y)
        assert result["status"] == "measured"

    def test_sufficient_data_mae_is_finite_positive(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(200)
        result = compute_rul_holdout_metrics(X, y)
        mae = result["holdout_mae_hours"]
        assert mae is not None
        assert math.isfinite(mae)
        assert mae > 0.0

    def test_sufficient_data_rmse_is_finite_positive(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(200)
        result = compute_rul_holdout_metrics(X, y)
        rmse = result["holdout_rmse_hours"]
        assert rmse is not None
        assert math.isfinite(rmse)
        assert rmse > 0.0

    def test_sufficient_data_rmse_ge_mae(self):
        """RMSE >= MAE always holds (Jensen's inequality)."""
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(200)
        result = compute_rul_holdout_metrics(X, y)
        assert result["holdout_rmse_hours"] >= result["holdout_mae_hours"]

    def test_sufficient_data_split_sizes_correct(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(200)
        result = compute_rul_holdout_metrics(X, y)
        n_train = result["holdout_n_train"]
        n_test = result["holdout_n_test"]
        assert n_train is not None and n_test is not None
        assert n_train + n_test == 200
        assert n_train >= 10 and n_test >= 10

    def test_sufficient_data_split_method(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(200)
        result = compute_rul_holdout_metrics(X, y)
        assert result["split_method"] == "temporal_80_20"

    def test_sufficient_data_reason_is_none(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(200)
        result = compute_rul_holdout_metrics(X, y)
        assert result["reason"] is None

    def test_insufficient_data_returns_none_metrics(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(50)  # below the 100-row minimum
        result = compute_rul_holdout_metrics(X, y)
        assert result["holdout_mae_hours"] is None
        assert result["holdout_rmse_hours"] is None
        assert result["holdout_n_train"] is None
        assert result["holdout_n_test"] is None

    def test_insufficient_data_status_and_reason(self):
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(30)
        result = compute_rul_holdout_metrics(X, y)
        assert result["status"] == "insufficient_samples"
        assert result["reason"] is not None
        assert len(result["reason"]) > 0

    def test_boundary_exactly_at_min_returns_measured(self):
        """Exactly 100 rows (the minimum) must yield a measured result."""
        from src.ml.model_card import compute_rul_holdout_metrics
        X, y = _make_rul_data(100)
        result = compute_rul_holdout_metrics(X, y)
        assert result["status"] == "measured"


# ---------------------------------------------------------------------------
# Integration: model card written by RULPredictor.train()
# ---------------------------------------------------------------------------

class TestRULPredictorModelCardMetrics:
    def test_model_card_has_measured_metrics_after_training(self, tmp_path):
        """After training on 200 rows the model card must report real numbers."""
        import json

        from src.ml.rul_predictor import RULPredictor

        p = RULPredictor("MC-TEST-01", beta=2.1, eta=720.0)
        X, y = _make_rul_data(200)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        p.train(X, y, feature_names)

        card_path = p._model_path.replace(".joblib", ".model_card.json")
        with open(card_path, encoding="utf-8") as fh:
            card = json.load(fh)

        metrics = card["metrics"]
        assert metrics["status"] == "measured", f"Expected 'measured', got: {metrics}"
        mae = metrics["holdout_mae_hours"]
        assert mae is not None
        assert math.isfinite(mae)
        assert mae > 0.0, f"holdout_mae_hours should be > 0, got {mae}"

    def test_model_card_metrics_rmse_present(self, tmp_path):
        import json

        from src.ml.rul_predictor import RULPredictor

        p = RULPredictor("MC-TEST-02", beta=2.1, eta=720.0)
        X, y = _make_rul_data(200)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        p.train(X, y, feature_names)

        card_path = p._model_path.replace(".joblib", ".model_card.json")
        with open(card_path, encoding="utf-8") as fh:
            card = json.load(fh)

        rmse = card["metrics"]["holdout_rmse_hours"]
        assert rmse is not None
        assert math.isfinite(rmse)
        assert rmse > 0.0

    def test_model_card_cold_start_degrades_gracefully(self, tmp_path):
        """Training is refused under 50 rows (RULPredictor guard).

        The model card written on a legacy load-without-card path must also
        degrade gracefully — verify _unmeasured_metrics for rul_predictor.
        """
        from src.ml.model_card import _unmeasured_metrics

        metrics = _unmeasured_metrics("rul_predictor")
        assert metrics["holdout_mae_hours"] is None
        assert metrics["holdout_rmse_hours"] is None
        assert metrics["status"] in ("insufficient_samples", "not_measured_in_current_simulation")
        assert metrics["reason"] is not None

    def test_model_card_anomaly_detector_unchanged(self, tmp_path):
        """_unmeasured_metrics for anomaly_detector must still return old shape."""
        from src.ml.model_card import _unmeasured_metrics

        metrics = _unmeasured_metrics("anomaly_detector")
        assert "roc_auc" in metrics
        assert metrics["status"] == "not_measured_in_current_simulation"
