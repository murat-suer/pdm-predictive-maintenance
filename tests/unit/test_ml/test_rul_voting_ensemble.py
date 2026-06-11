"""
Unit tests for RUL Voting Ensemble.

Tests the soft-voting ensemble of top-3 RUL regressors:
- RULVotingEnsemble initialization
- fit() and predict()
- Member names
- to_dict() serialization
- default_rul_ensemble_members() (3 members)
- build_default_rul_voting_ensemble()
- Edge cases: empty members, weights mismatch
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.ml.models.rul_voting_ensemble import (
    RULVotingEnsemble,
    build_default_rul_voting_ensemble,
    default_rul_ensemble_members,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_members():
    """Three simple estimators for testing (no heavy deps)."""
    return [
        ("model_a", DummyRegressor(strategy="mean")),
        ("model_b", DummyRegressor(strategy="mean")),
        ("model_c", DummyRegressor(strategy="mean")),
    ]


@pytest.fixture
def sample_data():
    """Simple regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = X @ np.array([1.0, 2.0, 0.5, -1.0, 0.3]) + rng.randn(50) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# RULVotingEnsemble initialization
# ---------------------------------------------------------------------------

class TestRULVotingEnsembleInit:
    def test_init_with_default_weights(self, simple_members):
        ensemble = RULVotingEnsemble(members=simple_members)
        assert len(ensemble.weights) == 3
        assert all(abs(w - 1.0 / 3) < 1e-9 for w in ensemble.weights)

    def test_init_with_custom_weights(self, simple_members):
        weights = [0.5, 0.3, 0.2]
        ensemble = RULVotingEnsemble(members=simple_members, weights=weights)
        assert ensemble.weights == weights

    def test_init_single_member(self):
        members = [("only_one", DummyRegressor())]
        ensemble = RULVotingEnsemble(members=members)
        assert ensemble.weights == [1.0]

    def test_init_empty_members_raises(self):
        with pytest.raises(ValueError, match="at least one member"):
            RULVotingEnsemble(members=[])

    def test_init_weights_length_mismatch_raises(self, simple_members):
        with pytest.raises(ValueError, match="weights length"):
            RULVotingEnsemble(members=simple_members, weights=[0.5, 0.5])

    def test_is_fitted_initially_false(self, simple_members):
        ensemble = RULVotingEnsemble(members=simple_members)
        assert ensemble.is_fitted is False


# ---------------------------------------------------------------------------
# fit() and predict()
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_sets_fitted_flag(self, simple_members, sample_data):
        X, y = sample_data
        ensemble = RULVotingEnsemble(members=simple_members)
        ensemble.fit(X, y)
        assert ensemble.is_fitted is True

    def test_fit_returns_self(self, simple_members, sample_data):
        X, y = sample_data
        ensemble = RULVotingEnsemble(members=simple_members)
        result = ensemble.fit(X, y)
        assert result is ensemble

    def test_predict_before_fit_raises(self, simple_members, sample_data):
        X, _ = sample_data
        ensemble = RULVotingEnsemble(members=simple_members)
        with pytest.raises(RuntimeError, match="must be fit before predict"):
            ensemble.predict(X)

    def test_predict_returns_correct_shape(self, simple_members, sample_data):
        X, y = sample_data
        ensemble = RULVotingEnsemble(members=simple_members)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert preds.shape == (50,)

    def test_predict_equal_weights_is_mean(self, simple_members, sample_data):
        """With equal weights, prediction should be the mean of member predictions."""
        X, y = sample_data
        ensemble = RULVotingEnsemble(members=simple_members)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        # DummyRegressor(strategy="mean") predicts the training mean for all X
        # All three members predict the same value, so the weighted average equals that value
        expected = np.full(50, y.mean())
        np.testing.assert_allclose(preds, expected, rtol=1e-6)

    def test_predict_weighted(self, sample_data):
        """Test that weights affect the output."""
        X, y = sample_data
        # Use two different models with different strategies
        members = [
            ("mean_model", DummyRegressor(strategy="mean")),
            ("median_model", DummyRegressor(strategy="median")),
        ]
        # Heavily weight the median model
        ensemble = RULVotingEnsemble(members=members, weights=[0.0, 1.0])
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        expected = np.full(50, np.median(y))
        np.testing.assert_allclose(preds, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# member_names()
# ---------------------------------------------------------------------------

class TestMemberNames:
    def test_member_names(self, simple_members):
        ensemble = RULVotingEnsemble(members=simple_members)
        names = ensemble.member_names()
        assert names == ["model_a", "model_b", "model_c"]


# ---------------------------------------------------------------------------
# to_dict() serialization
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_default_weights(self, simple_members):
        ensemble = RULVotingEnsemble(members=simple_members)
        d = ensemble.to_dict()
        assert d["members"] == ["model_a", "model_b", "model_c"]
        assert len(d["weights"]) == 3
        assert all(abs(w - 1.0 / 3) < 1e-9 for w in d["weights"])

    def test_to_dict_custom_weights(self, simple_members):
        weights = [0.5, 0.3, 0.2]
        ensemble = RULVotingEnsemble(members=simple_members, weights=weights)
        d = ensemble.to_dict()
        assert d["weights"] == weights

    def test_to_dict_weights_are_floats(self, simple_members):
        ensemble = RULVotingEnsemble(members=simple_members)
        d = ensemble.to_dict()
        assert all(isinstance(w, float) for w in d["weights"])


# ---------------------------------------------------------------------------
# default_rul_ensemble_members()
# ---------------------------------------------------------------------------

class TestDefaultRULEnsembleMembers:
    def test_returns_three_members(self):
        members = default_rul_ensemble_members()
        assert len(members) == 3

    def test_members_are_name_estimator_tuples(self):
        members = default_rul_ensemble_members()
        for name, estimator in members:
            assert isinstance(name, str)
            assert hasattr(estimator, "fit")
            assert hasattr(estimator, "predict")

    def test_first_member_is_gradient_boosting(self):
        members = default_rul_ensemble_members()
        assert members[0][0] == "GradientBoosting"

    def test_second_member_is_xgboost(self):
        members = default_rul_ensemble_members()
        assert members[1][0] == "XGBoost"

    def test_third_member_is_lightgbm_or_random_forest(self):
        members = default_rul_ensemble_members()
        assert members[2][0] in ("LightGBM", "RandomForest")


# ---------------------------------------------------------------------------
# build_default_rul_voting_ensemble()
# ---------------------------------------------------------------------------

class TestBuildDefaultRULVotingEnsemble:
    def test_returns_ensemble(self):
        ensemble = build_default_rul_voting_ensemble()
        assert isinstance(ensemble, RULVotingEnsemble)

    def test_has_three_members(self):
        ensemble = build_default_rul_voting_ensemble()
        assert len(ensemble.members) == 3

    def test_default_equal_weights(self):
        ensemble = build_default_rul_voting_ensemble()
        assert all(abs(w - 1.0 / 3) < 1e-9 for w in ensemble.weights)

    def test_custom_weights(self):
        weights = [0.5, 0.3, 0.2]
        ensemble = build_default_rul_voting_ensemble(weights=weights)
        assert ensemble.weights == weights

    def test_fit_predict_integration(self, sample_data):
        """Full integration test: build, fit, predict."""
        X, y = sample_data
        ensemble = build_default_rul_voting_ensemble()
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert preds.shape == (50,)
        assert not np.any(np.isnan(preds))
