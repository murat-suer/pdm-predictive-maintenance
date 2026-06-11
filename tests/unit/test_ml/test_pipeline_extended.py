"""Tests for extended training pipeline with PINNs integration."""
import numpy as np
import pytest
import torch

from src.ml.pipeline import (
    PINN_MIN_SAMPLES,
    TrainingMetrics,
    select_model_type,
    train_model,
    train_pinn_model,
    train_xgboost_model,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def small_dataset():
    """Small dataset suitable for XGBoost (< PINN_MIN_SAMPLES)."""
    rng = np.random.RandomState(42)
    n_samples = 100
    n_features = 6
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.rand(n_samples).astype(np.float32) * 200  # RUL values
    timestamps = np.linspace(0, 100, n_samples).astype(np.float32)
    return X, y, timestamps


@pytest.fixture
def large_dataset():
    """Larger dataset suitable for PINN (>= PINN_MIN_SAMPLES)."""
    rng = np.random.RandomState(42)
    n_samples = 600
    n_features = 6
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.rand(n_samples).astype(np.float32) * 200
    timestamps = np.linspace(0, 500, n_samples).astype(np.float32)
    return X, y, timestamps


@pytest.fixture
def tiny_dataset():
    """Very small dataset for edge case testing."""
    rng = np.random.RandomState(42)
    n_samples = 10
    n_features = 4
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.rand(n_samples).astype(np.float32) * 100
    timestamps = np.linspace(0, 50, n_samples).astype(np.float32)
    return X, y, timestamps


# ─── Test select_model_type ──────────────────────────────────────────────────


class TestSelectModelType:
    def test_small_dataset_selects_xgboost(self):
        """Datasets below threshold should select XGBoost."""
        assert select_model_type(100) == "xgboost"
        assert select_model_type(499) == "xgboost"

    def test_large_dataset_selects_pinn(self):
        """Datasets at or above threshold should select PINN."""
        assert select_model_type(500) == "pinn"
        assert select_model_type(1000) == "pinn"
        assert select_model_type(10000) == "pinn"

    def test_custom_threshold(self):
        """Custom threshold should override default."""
        assert select_model_type(50, threshold=50) == "pinn"
        assert select_model_type(49, threshold=50) == "xgboost"

    def test_zero_samples(self):
        """Zero samples should select XGBoost."""
        assert select_model_type(0) == "xgboost"

    def test_exact_threshold(self):
        """Exactly at threshold should select PINN."""
        assert select_model_type(PINN_MIN_SAMPLES) == "pinn"


# ─── Test TrainingMetrics ────────────────────────────────────────────────────


class TestTrainingMetrics:
    def test_creation(self):
        """TrainingMetrics should be creatable with required fields."""
        metrics = TrainingMetrics(
            model_type="pinn",
            n_samples=100,
            n_features=6,
        )
        assert metrics.model_type == "pinn"
        assert metrics.n_samples == 100
        assert metrics.n_features == 6
        assert metrics.n_epochs == 0
        assert metrics.final_loss == 0.0

    def test_to_dict(self):
        """to_dict should return JSON-serializable dict."""
        metrics = TrainingMetrics(
            model_type="xgboost",
            n_samples=500,
            n_features=8,
            n_epochs=200,
            final_loss=0.05,
            final_data_loss=0.05,
            final_physics_loss=0.0,
            loss_history=[0.1, 0.08, 0.05],
        )
        d = metrics.to_dict()
        assert d["model_type"] == "xgboost"
        assert d["n_samples"] == 500
        assert d["n_epochs"] == 200
        assert d["final_loss"] == 0.05
        assert d["loss_history_length"] == 3

    def test_default_values(self):
        """Default values should be sensible."""
        metrics = TrainingMetrics(model_type="pinn", n_samples=10, n_features=3)
        assert metrics.loss_history == []
        assert metrics.extra == {}
        assert metrics.final_physics_loss == 0.0


# ─── Test train_pinn_model ───────────────────────────────────────────────────


class TestTrainPinnModel:
    def test_basic_training(self, small_dataset):
        """PINN should train without errors and return model + metrics."""
        X, y, timestamps = small_dataset
        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=5,  # Keep fast for testing
        )
        assert model is not None
        assert metrics.model_type == "pinn"
        assert metrics.n_samples == 100
        assert metrics.n_features == 6
        assert metrics.n_epochs == 5
        assert len(metrics.loss_history) == 5
        assert metrics.final_loss > 0

    def test_loss_decreases(self, small_dataset):
        """Loss should generally decrease during training."""
        X, y, timestamps = small_dataset
        _, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=20,
        )
        # Final loss should be less than initial loss
        assert metrics.loss_history[-1] < metrics.loss_history[0]

    def test_model_produces_predictions(self, small_dataset):
        """Trained model should produce predictions."""
        X, y, timestamps = small_dataset
        model, _ = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
        )
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X[:5]), torch.tensor(timestamps[:5]))
        assert preds.shape == (5,)

    def test_physics_loss_tracked(self, small_dataset):
        """Physics loss should be tracked in metrics."""
        X, y, timestamps = small_dataset
        _, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
            lambda_physics=0.5,
        )
        assert metrics.final_physics_loss >= 0
        assert metrics.extra["lambda_physics"] == 0.5
        assert metrics.extra["eta"] == 500.0
        assert metrics.extra["beta"] == 2.0

    def test_custom_hidden_dim(self, small_dataset):
        """Custom hidden dimension should be respected."""
        X, y, timestamps = small_dataset
        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=2,
            hidden_dim=32,
        )
        assert metrics.extra["hidden_dim"] == 32

    def test_zero_lambda_physics(self, small_dataset):
        """With lambda_physics=0, should still train (pure data loss)."""
        X, y, timestamps = small_dataset
        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
            lambda_physics=0.0,
        )
        assert metrics.final_loss > 0


# ─── Test train_xgboost_model ────────────────────────────────────────────────


class TestTrainXGBoostModel:
    def test_basic_training(self, small_dataset):
        """XGBoost should train without errors."""
        X, y, _ = small_dataset
        model, metrics = train_xgboost_model(
            X_train=X,
            y_train=y,
            n_estimators=10,
        )
        assert model is not None
        assert metrics.model_type == "xgboost"
        assert metrics.n_samples == 100
        assert metrics.n_features == 6
        assert metrics.final_loss >= 0

    def test_predictions(self, small_dataset):
        """XGBoost should produce predictions."""
        X, y, _ = small_dataset
        model, _ = train_xgboost_model(X_train=X, y_train=y, n_estimators=10)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_custom_params(self, small_dataset):
        """Custom parameters should be passed through."""
        X, y, _ = small_dataset
        _, metrics = train_xgboost_model(
            X_train=X,
            y_train=y,
            n_estimators=50,
            max_depth=3,
        )
        assert metrics.extra["params"]["n_estimators"] == 50
        assert metrics.extra["params"]["max_depth"] == 3


# ─── Test train_model (unified interface) ────────────────────────────────────


class TestTrainModel:
    def test_auto_select_xgboost_small(self, small_dataset):
        """Auto-selection should pick XGBoost for small datasets."""
        X, y, timestamps = small_dataset
        model, metrics = train_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
        )
        assert metrics.model_type == "xgboost"

    def test_auto_select_pinn_large(self, large_dataset):
        """Auto-selection should pick PINN for large datasets."""
        X, y, timestamps = large_dataset
        model, metrics = train_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
        )
        assert metrics.model_type == "pinn"

    def test_explicit_pinn(self, small_dataset):
        """Explicit model_type='pinn' should use PINN regardless of size."""
        X, y, timestamps = small_dataset
        model, metrics = train_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            model_type="pinn",
            n_epochs=3,
        )
        assert metrics.model_type == "pinn"

    def test_explicit_xgboost(self, large_dataset):
        """Explicit model_type='xgboost' should use XGBoost regardless of size."""
        X, y, timestamps = large_dataset
        model, metrics = train_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            model_type="xgboost",
            n_estimators=10,
        )
        assert metrics.model_type == "xgboost"

    def test_pinn_without_timestamps_raises(self, small_dataset):
        """PINN without timestamps should raise ValueError."""
        X, y, _ = small_dataset
        with pytest.raises(ValueError, match="timestamps required"):
            train_model(
                X_train=X,
                y_train=y,
                timestamps=None,
                model_type="pinn",
                eta=500.0,
                beta=2.0,
            )

    def test_unknown_model_type_raises(self, small_dataset):
        """Unknown model_type should raise ValueError."""
        X, y, timestamps = small_dataset
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_model(
                X_train=X,
                y_train=y,
                timestamps=timestamps,
                model_type="random_forest",
            )


# ─── Test edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_tiny_dataset_xgboost(self, tiny_dataset):
        """XGBoost should handle very small datasets."""
        X, y, _ = tiny_dataset
        model, metrics = train_xgboost_model(
            X_train=X,
            y_train=y,
            n_estimators=5,
        )
        assert metrics.n_samples == 10
        assert metrics.model_type == "xgboost"

    def test_tiny_dataset_pinn(self, tiny_dataset):
        """PINN should handle very small datasets (even if not recommended)."""
        X, y, timestamps = tiny_dataset
        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
        )
        assert metrics.n_samples == 10
        assert metrics.model_type == "pinn"

    def test_single_feature(self):
        """Model should work with single feature."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 1).astype(np.float32)
        y = rng.rand(50).astype(np.float32) * 100
        timestamps = np.linspace(0, 50, 50).astype(np.float32)

        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
        )
        assert metrics.n_features == 1

    def test_many_features(self):
        """Model should work with many features."""
        rng = np.random.RandomState(42)
        n_features = 50
        X = rng.randn(100, n_features).astype(np.float32)
        y = rng.rand(100).astype(np.float32) * 100
        timestamps = np.linspace(0, 100, 100).astype(np.float32)

        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
        )
        assert metrics.n_features == n_features

    def test_high_lambda_physics(self, small_dataset):
        """High physics loss weight should still train."""
        X, y, timestamps = small_dataset
        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=500.0,
            beta=2.0,
            n_epochs=3,
            lambda_physics=10.0,
        )
        assert metrics.final_loss > 0
        assert metrics.extra["lambda_physics"] == 10.0

    def test_different_weibull_params(self, small_dataset):
        """Different Weibull parameters should work."""
        X, y, timestamps = small_dataset
        # High beta (more predictable failure)
        model, metrics = train_pinn_model(
            X_train=X,
            y_train=y,
            timestamps=timestamps,
            eta=1000.0,
            beta=5.0,
            n_epochs=3,
        )
        assert metrics.extra["eta"] == 1000.0
        assert metrics.extra["beta"] == 5.0

    def test_model_selection_boundary(self):
        """Test exact boundary of model selection."""
        assert select_model_type(PINN_MIN_SAMPLES - 1) == "xgboost"
        assert select_model_type(PINN_MIN_SAMPLES) == "pinn"
        assert select_model_type(PINN_MIN_SAMPLES + 1) == "pinn"
