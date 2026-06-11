"""
Tests for the evaluation framework modules:
- leakage_test
- permutation_importance
- rul_cv
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

# =============================================================================
# leakage_test tests
# =============================================================================


class TestLeakageTest:
    """Tests for src.ml.evaluation.leakage_test"""

    def test_import(self):
        """Module imports successfully."""
        from src.ml.evaluation.leakage_test import (
            LeakageReport,
            LeakageVerdict,
            run_leakage_check,
        )
        assert run_leakage_check is not None
        assert LeakageReport is not None
        assert LeakageVerdict is not None

    def test_clean_pipeline_passes(self):
        """A clean pipeline with no leakage passes all checks."""
        from src.ml.evaluation.leakage_test import run_leakage_check

        rng = np.random.default_rng(42)
        n = 200
        X_train = rng.normal(0, 1, (n, 3))
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = rng.normal(0, 1, (50, 3))
        y_test = (X_test[:, 0] > 0).astype(int)

        def model_factory():
            return LogisticRegression(random_state=42)

        report = run_leakage_check(
            model_factory,
            X_train,
            y_train,
            X_test,
            y_test,
            n_permutations=3,
            random_state=42,
        )

        assert report.pr_auc_real > 0.5
        assert report.baseline >= 0.0
        assert report.baseline <= 1.0
        assert report.n_permutations == 3
        assert len(report.permutation_pr_aucs) == 3
        assert len(report.verdicts) == 3  # permutation, temporal, feature_corr

    def test_leakage_detected_with_target_feature(self):
        """A feature perfectly correlated with target is detected."""
        from src.ml.evaluation.leakage_test import run_leakage_check

        rng = np.random.default_rng(42)
        n = 100
        y_train = rng.integers(0, 2, n)
        # Feature 0 is a perfect copy of the target
        X_train = np.column_stack([y_train.astype(float), rng.normal(0, 1, (n, 2))])
        X_test = rng.normal(0, 1, (50, 3))
        y_test = rng.integers(0, 2, 50)

        def model_factory():
            return LogisticRegression(random_state=42)

        report = run_leakage_check(
            model_factory,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_corr_limit=0.99,
            n_permutations=2,
            random_state=42,
        )

        # Should detect the suspect feature
        assert len(report.suspect_features) > 0
        assert report.suspect_features[0][0] == "f0"
        assert abs(report.suspect_features[0][1]) >= 0.99

    def test_temporal_order_check_passes(self):
        """Temporal order check passes when train precedes test."""
        from src.ml.evaluation.leakage_test import run_leakage_check

        rng = np.random.default_rng(42)
        n = 100
        X_train = rng.normal(0, 1, (n, 2))
        y_train = rng.integers(0, 2, n)
        X_test = rng.normal(0, 1, (30, 2))
        y_test = rng.integers(0, 2, 30)
        train_ts = np.arange(n)
        test_ts = np.arange(n, n + 30)

        def model_factory():
            return LogisticRegression(random_state=42)

        report = run_leakage_check(
            model_factory,
            X_train,
            y_train,
            X_test,
            y_test,
            train_timestamps=train_ts,
            test_timestamps=test_ts,
            n_permutations=2,
            random_state=42,
        )

        # Find the temporal verdict
        temporal_verdict = [v for v in report.verdicts if v.name == "temporal_order"][0]
        assert temporal_verdict.passed == True

    def test_temporal_order_check_fails(self):
        """Temporal order check fails when train overlaps test."""
        from src.ml.evaluation.leakage_test import run_leakage_check

        rng = np.random.default_rng(42)
        n = 100
        X_train = rng.normal(0, 1, (n, 2))
        y_train = rng.integers(0, 2, n)
        X_test = rng.normal(0, 1, (30, 2))
        y_test = rng.integers(0, 2, 30)
        # Overlapping timestamps
        train_ts = np.arange(n)
        test_ts = np.arange(50, 80)

        def model_factory():
            return LogisticRegression(random_state=42)

        report = run_leakage_check(
            model_factory,
            X_train,
            y_train,
            X_test,
            y_test,
            train_timestamps=train_ts,
            test_timestamps=test_ts,
            n_permutations=2,
            random_state=42,
        )

        temporal_verdict = [v for v in report.verdicts if v.name == "temporal_order"][0]
        assert temporal_verdict.passed == False

    def test_report_summary(self):
        """Report summary is a non-empty string."""
        from src.ml.evaluation.leakage_test import run_leakage_check

        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (100, 2))
        y_train = rng.integers(0, 2, 100)
        X_test = rng.normal(0, 1, (30, 2))
        y_test = rng.integers(0, 2, 30)

        def model_factory():
            return LogisticRegression(random_state=42)

        report = run_leakage_check(
            model_factory, X_train, y_train, X_test, y_test, n_permutations=2
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "verdict:" in summary

    def test_invalid_n_permutations_raises(self):
        """n_permutations < 1 raises ValueError."""
        from src.ml.evaluation.leakage_test import run_leakage_check

        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (50, 2))
        y_train = rng.integers(0, 2, 50)
        X_test = rng.normal(0, 1, (20, 2))
        y_test = rng.integers(0, 2, 20)

        def model_factory():
            return LogisticRegression(random_state=42)

        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            run_leakage_check(
                model_factory, X_train, y_train, X_test, y_test, n_permutations=0
            )


# =============================================================================
# permutation_importance tests
# =============================================================================


class TestPermutationImportance:
    """Tests for src.ml.evaluation.permutation_importance"""

    def test_import(self):
        """Module imports successfully."""
        from src.ml.evaluation.permutation_importance import (
            PermutationImportanceResult,
            permutation_importance,
        )
        assert permutation_importance is not None
        assert PermutationImportanceResult is not None

    def test_rul_importance(self, tmp_path):
        """Permutation importance works for RUL regression."""
        from src.ml.evaluation.permutation_importance import permutation_importance

        rng = np.random.default_rng(42)
        n = 200
        feature_names = ["vibration", "temp", "pressure"]
        X = rng.normal(0, 1, (n, 3))
        # Strong signal from first feature
        y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.1 * rng.normal(0, 1, n)

        result = permutation_importance(
            "GradientBoosting",
            X,
            y,
            feature_names,
            family="rul",
            n_repeats=3,
            random_state=42,
            top_n=3,
            artifact_dir=str(tmp_path),
        )

        assert result.model_name == "GradientBoosting"
        assert result.family == "rul"
        assert len(result.feature_names) == 3
        assert len(result.importances_mean) == 3
        assert len(result.importances_std) == 3
        assert result.n_repeats == 3
        assert len(result.top_features) == 3
        # Top feature should be "vibration" (coefficient 2.0)
        assert result.top_features[0]["feature"] == "vibration"
        assert result.top_features[0]["importance_mean"] > 0

    def test_to_dict(self, tmp_path):
        """Result to_dict returns expected keys."""
        from src.ml.evaluation.permutation_importance import permutation_importance

        rng = np.random.default_rng(42)
        n = 100
        feature_names = ["a", "b"]
        X = rng.normal(0, 1, (n, 2))
        y = X[:, 0] + rng.normal(0, 0.1, n)

        result = permutation_importance(
            "RandomForest",
            X,
            y,
            feature_names,
            family="rul",
            n_repeats=2,
            artifact_dir=str(tmp_path),
        )

        d = result.to_dict()
        assert "model_name" in d
        assert "family" in d
        assert "n_features" in d
        assert "top_features" in d
        assert "importances_mean" in d
        assert "importances_std" in d
        assert d["n_features"] == 2

    def test_json_artifact_written(self, tmp_path):
        """JSON artifact is written to disk."""
        import json
        import os

        from src.ml.evaluation.permutation_importance import permutation_importance

        rng = np.random.default_rng(42)
        n = 100
        feature_names = ["a", "b"]
        X = rng.normal(0, 1, (n, 2))
        y = X[:, 0] + rng.normal(0, 0.1, n)

        result = permutation_importance(
            "RandomForest",
            X,
            y,
            feature_names,
            family="rul",
            n_repeats=2,
            artifact_dir=str(tmp_path),
        )

        json_path = os.path.join(str(tmp_path), "permutation_importance.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data["model_name"] == "RandomForest"
        assert data["n_features"] == 2

    def test_feature_mismatch_raises(self, tmp_path):
        """Mismatched feature_names length raises ValueError."""
        from src.ml.evaluation.permutation_importance import permutation_importance

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        y = rng.normal(0, 1, 50)

        with pytest.raises(ValueError, match="columns but feature_names"):
            permutation_importance(
                "RandomForest",
                X,
                y,
                ["a", "b"],  # wrong length
                family="rul",
                artifact_dir=str(tmp_path),
            )


# =============================================================================
# rul_cv tests
# =============================================================================


class TestRulCV:
    """Tests for src.ml.evaluation.rul_cv"""

    def test_import(self):
        """Module imports successfully."""
        from src.ml.evaluation.rul_cv import (
            VALID_STRATEGIES,
            CVFoldResult,
            CVStrategyReport,
            cross_validate_rul,
        )
        assert cross_validate_rul is not None
        assert CVStrategyReport is not None
        assert CVFoldResult is not None
        assert "per_machine" in VALID_STRATEGIES
        assert "leave_one_machine_out" in VALID_STRATEGIES
        assert "k_fold" in VALID_STRATEGIES

    def test_per_machine_strategy(self):
        """per_machine strategy creates one fold per machine."""
        from src.ml.evaluation.rul_cv import cross_validate_rul

        rng = np.random.default_rng(42)
        n_per_machine = 100
        n_machines = 3
        n_total = n_per_machine * n_machines
        X = rng.normal(0, 1, (n_total, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.1, n_total)
        machine_ids = np.array(
            [f"M{i // n_per_machine}" for i in range(n_total)]
        )

        report = cross_validate_rul(
            "RandomForest",
            X,
            y,
            machine_ids,
            strategy="per_machine",
            min_train_rows=10,
            min_test_rows=5,
        )

        assert report.strategy == "per_machine"
        assert report.model_name == "RandomForest"
        assert report.metric == "mae"
        assert len(report.fold_results) == n_machines
        assert not np.isnan(report.mean_test_metric)
        assert report.mean_test_metric >= 0

    def test_leave_one_machine_out(self):
        """leave_one_machine_out creates N folds (one per held-out machine)."""
        from src.ml.evaluation.rul_cv import cross_validate_rul

        rng = np.random.default_rng(42)
        n_per_machine = 100
        n_machines = 3
        n_total = n_per_machine * n_machines
        X = rng.normal(0, 1, (n_total, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.1, n_total)
        machine_ids = np.array(
            [f"M{i // n_per_machine}" for i in range(n_total)]
        )

        report = cross_validate_rul(
            "RandomForest",
            X,
            y,
            machine_ids,
            strategy="leave_one_machine_out",
            min_train_rows=10,
            min_test_rows=5,
        )

        assert report.strategy == "leave_one_machine_out"
        assert len(report.fold_results) == n_machines
        # Each fold has one test machine
        for fold in report.fold_results:
            assert len(fold.test_machines) == 1
            assert len(fold.train_machines) == n_machines - 1

    def test_k_fold_strategy(self):
        """k_fold strategy creates k folds."""
        from src.ml.evaluation.rul_cv import cross_validate_rul

        rng = np.random.default_rng(42)
        n_per_machine = 100
        n_machines = 6
        n_total = n_per_machine * n_machines
        X = rng.normal(0, 1, (n_total, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.1, n_total)
        machine_ids = np.array(
            [f"M{i // n_per_machine}" for i in range(n_total)]
        )

        report = cross_validate_rul(
            "RandomForest",
            X,
            y,
            machine_ids,
            strategy="k_fold",
            k=3,
            seed=42,
            min_train_rows=10,
            min_test_rows=5,
        )

        assert report.strategy == "k_fold"
        assert len(report.fold_results) == 3

    def test_to_dict(self):
        """Report to_dict returns expected structure."""
        from src.ml.evaluation.rul_cv import cross_validate_rul

        rng = np.random.default_rng(42)
        n_per_machine = 80
        n_machines = 2
        n_total = n_per_machine * n_machines
        X = rng.normal(0, 1, (n_total, 2))
        y = X[:, 0] + rng.normal(0, 0.1, n_total)
        machine_ids = np.array(
            [f"M{i // n_per_machine}" for i in range(n_total)]
        )

        report = cross_validate_rul(
            "RandomForest",
            X,
            y,
            machine_ids,
            strategy="leave_one_machine_out",
            min_train_rows=10,
            min_test_rows=5,
        )

        d = report.to_dict()
        assert "strategy" in d
        assert "model_name" in d
        assert "metric" in d
        assert "n_folds" in d
        assert "mean_test_metric" in d
        assert "std_test_metric" in d
        assert "fold_results" in d
        assert d["n_folds"] == 2

    def test_insufficient_rows_skipped(self):
        """Folds with too few rows are skipped."""
        from src.ml.evaluation.rul_cv import cross_validate_rul

        rng = np.random.default_rng(42)
        # Very few rows per machine
        n_per_machine = 5
        n_machines = 2
        n_total = n_per_machine * n_machines
        X = rng.normal(0, 1, (n_total, 2))
        y = X[:, 0] + rng.normal(0, 0.1, n_total)
        machine_ids = np.array(
            [f"M{i // n_per_machine}" for i in range(n_total)]
        )

        report = cross_validate_rul(
            "RandomForest",
            X,
            y,
            machine_ids,
            strategy="per_machine",
            min_train_rows=50,  # impossible to meet
            min_test_rows=50,
        )

        assert report.n_skipped_folds > 0
        assert np.isnan(report.mean_test_metric)

    def test_invalid_strategy_raises(self):
        """Invalid strategy raises ValueError."""
        from src.ml.evaluation.rul_cv import cross_validate_rul

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 2))
        y = rng.normal(0, 1, 50)
        machine_ids = np.array(["M0"] * 25 + ["M1"] * 25)

        with pytest.raises(ValueError, match="strategy must be one of"):
            cross_validate_rul(
                "RandomForest",
                X,
                y,
                machine_ids,
                strategy="invalid_strategy",
            )

    def test_build_folds_deterministic(self):
        """k_fold with same seed produces same folds."""
        from src.ml.evaluation.rul_cv import _build_folds

        machine_ids = np.array(["M0"] * 10 + ["M1"] * 10 + ["M2"] * 10 + ["M3"] * 10)

        folds1 = _build_folds(machine_ids, "k_fold", k=2, seed=42)
        folds2 = _build_folds(machine_ids, "k_fold", k=2, seed=42)

        assert len(folds1) == len(folds2)
        for (t1, te1, tm1, tem1), (t2, te2, tm2, tem2) in zip(folds1, folds2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(te1, te2)
            assert tm1 == tm2
            assert tem1 == tem2
