"""
src/ml/evaluation/model_comparison.py
======================================
Multi-model benchmark runner for anomaly detection and RUL regression.

The benchmark is the user-facing entry point that selects the production
model. Inputs are a list of :class:`ModelSpec` (factory + hyperparameters
+ contract) and a temporal split. Outputs are:

* a :class:`BenchmarkResult` per model containing train / val / test
  metrics, fit and predict wall-clock time, and serialized model size;
* a comparison table ordered by validation score with a configurable
  arbiter on test;
* an :class:`OverfitFlag` highlighting any model whose validation-test
  gap exceeds ``overfit_tolerance`` (a leakage / overfit signal).

The two pre-configured ladders, :func:`default_anomaly_models` and
:func:`default_rul_models`, are the literature-supported model panels
the user explicitly asked for in the Phase 2A brief. Adding a model
means appending another :class:`ModelSpec`; nothing else changes.

References
----------
* Liu, Ting & Zhou (2008). "Isolation Forest." ICDM —
  ``IsolationForest`` baseline.
* Breunig et al. (2000). "LOF: Identifying Density-Based Local
  Outliers." SIGMOD — ``LocalOutlierFactor``.
* Schölkopf et al. (2001). "Estimating the Support of a
  High-Dimensional Distribution." Neural Computation 13(7) —
  ``OneClassSVM``.
* Rousseeuw & Van Driessen (1999). "A Fast Algorithm for the Minimum
  Covariance Determinant Estimator." Technometrics 41(3) —
  ``EllipticEnvelope``.
* Hinton & Salakhutdinov (2006). "Reducing the Dimensionality of Data
  with Neural Networks." Science 313(5786) — auto-encoder bottleneck.
* Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System."
  KDD — ``XGBRegressor``.
* Breiman (2001). "Random Forests." Machine Learning 45(1) —
  ``RandomForestRegressor``.
* Friedman (2001). "Greedy Function Approximation: A Gradient Boosting
  Machine." Annals of Statistics 29(5) — ``GradientBoostingRegressor``.
* Weibull (1951). "A Statistical Distribution Function of Wide
  Applicability." J. Appl. Mech. 18(3) — parametric Weibull baseline.
"""

from __future__ import annotations

import io
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from src.ml.evaluation.metrics import (
    best_f1_threshold,
    f1_at_threshold,
    mae,
    pr_auc,
)

DEFAULT_OVERFIT_TOLERANCE = 0.05


def _safe_pickle_size(obj) -> int:
    """Approximate the serialized model size in bytes via ``pickle``.

    Returns ``-1`` for any obj that ``pickle`` refuses (e.g. local
    classes defined inside test bodies or stateful objects with thread
    handles). The benchmark reports the size as diagnostic data only —
    a failed measurement must never crash the run.
    """

    buf = io.BytesIO()
    try:
        pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return -1
    return buf.tell()


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a candidate model for benchmark comparison.

    Attributes
    ----------
    name
        Human-readable, file-safe identifier (used as the row label).
    factory
        Zero-arg callable returning a fresh, *unfitted* estimator. A
        factory (rather than an instance) is required so each fold gets
        an independent estimator and there is no accidental state
        carry-over.
    family
        ``"anomaly"`` or ``"rul"`` — determines which scoring path is
        used.
    description
        One-sentence rationale; included in the report. Keeps the
        provenance of every panel member traceable.
    hyperparameters
        Dict of the values used by ``factory``; reproduced in the
        report verbatim. Hyperparameters MUST come from cited defaults
        or be drawn from a documented distribution — never hand-tuned.
    needs_fit_y
        ``True`` for supervised models (RUL regressors). ``False`` for
        unsupervised anomaly detectors that call ``.fit(X)`` only.
    """

    name: str
    factory: Callable
    family: str
    description: str
    hyperparameters: dict
    needs_fit_y: bool = False


@dataclass
class BenchmarkResult:
    """Outcome of evaluating one :class:`ModelSpec` on one split."""

    spec: ModelSpec
    train_score: float
    val_score: float
    test_score: float
    val_secondary: float
    test_secondary: float
    train_secondary: float | None = None
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    model_size_bytes: int = -1
    threshold: float | None = None
    extra: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class OverfitFlag:
    """Marker emitted when a model's val/test gap looks suspicious."""

    name: str
    val_score: float
    test_score: float
    gap: float
    tolerance: float


@dataclass
class BenchmarkReport:
    """Aggregated benchmark output."""

    family: str
    results: list[BenchmarkResult]
    winner: BenchmarkResult | None
    overfit_flags: list[OverfitFlag]
    metric_name: str
    secondary_metric_name: str

    def to_table(self) -> list[dict]:
        """Flatten results to a list of dicts (one per model) for tabular display."""

        rows = []
        for r in self.results:
            rows.append(
                {
                    "name": r.spec.name,
                    "family": r.spec.family,
                    self.metric_name + "_train": round(r.train_score, 4),
                    self.metric_name + "_val": round(r.val_score, 4),
                    self.metric_name + "_test": round(r.test_score, 4),
                    self.secondary_metric_name + "_val": round(r.val_secondary, 4),
                    self.secondary_metric_name + "_test": round(r.test_secondary, 4),
                    self.secondary_metric_name + "_train": round(r.train_secondary, 4) if r.train_secondary is not None else None,
                    "fit_seconds": round(r.fit_seconds, 4),
                    "predict_seconds": round(r.predict_seconds, 4),
                    "model_size_bytes": r.model_size_bytes,
                    "threshold": (None if r.threshold is None else round(r.threshold, 4)),
                    "error": r.error,
                }
            )
        return rows


class ModelBenchmark:
    """Run a panel of candidate models through a temporal split and rank them.

    The benchmark is intentionally tiny — it does only what the Phase 2A
    brief asks for. Anything stateful (hyperparameter search, ensembling,
    cross-validation aggregation) is *out of scope*; the temporal split
    is the only validation strategy supported.

    Parameters
    ----------
    family
        ``"anomaly"`` or ``"rul"``. Selects the metric (PR-AUC vs MAE)
        and the calling convention for the estimators.
    overfit_tolerance
        Maximum tolerated absolute gap between validation and test
        scores. A model whose ``|val − test| > overfit_tolerance`` is
        flagged via :class:`OverfitFlag` and excluded from the winner
        selection (the brief calls this an overfit signal). Default
        ``0.05`` matches the Phase 2A specification.
    """

    def __init__(
        self,
        family: str = "anomaly",
        overfit_tolerance: float = DEFAULT_OVERFIT_TOLERANCE,
    ):
        if family not in ("anomaly", "rul"):
            raise ValueError("family must be 'anomaly' or 'rul'")
        if overfit_tolerance < 0:
            raise ValueError("overfit_tolerance must be >= 0")
        self.family = family
        self.overfit_tolerance = float(overfit_tolerance)

    def run(
        self,
        specs: Sequence[ModelSpec],
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) -> BenchmarkReport:
        """Train each spec on the train split, score on val and test."""

        X_train = np.asarray(X_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        y_test = np.asarray(y_test)

        results: list[BenchmarkResult] = []
        for spec in specs:
            if spec.family != self.family:
                raise ValueError(
                    f"spec {spec.name!r} is family={spec.family!r}, "
                    f"benchmark expected {self.family!r}"
                )
            try:
                result = self._evaluate_one(
                    spec, X_train, y_train, X_val, y_val, X_test, y_test
                )
            except Exception as exc:
                result = BenchmarkResult(
                    spec=spec,
                    train_score=float("nan"),
                    val_score=float("nan"),
                    test_score=float("nan"),
                    val_secondary=float("nan"),
                    test_secondary=float("nan"),
                    fit_seconds=0.0,
                    predict_seconds=0.0,
                    model_size_bytes=-1,
                    threshold=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)

        overfit_flags = self._overfit_flags(results)
        winner = self._select_winner(results, overfit_flags)
        primary, secondary = self._metric_names()
        return BenchmarkReport(
            family=self.family,
            results=results,
            winner=winner,
            overfit_flags=overfit_flags,
            metric_name=primary,
            secondary_metric_name=secondary,
        )

    def _metric_names(self) -> tuple[str, str]:
        if self.family == "anomaly":
            return "pr_auc", "f1"
        return "neg_mae", "mae"

    def _evaluate_one(
        self,
        spec: ModelSpec,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) -> BenchmarkResult:
        if self.family == "anomaly":
            return self._evaluate_anomaly(spec, X_train, y_train, X_val, y_val, X_test, y_test)
        return self._evaluate_rul(spec, X_train, y_train, X_val, y_val, X_test, y_test)

    def _evaluate_anomaly(
        self,
        spec: ModelSpec,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) -> BenchmarkResult:
        model = spec.factory()
        t0 = time.perf_counter()
        if spec.needs_fit_y:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)
        fit_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        score_train = _anomaly_score(model, X_train)
        score_val = _anomaly_score(model, X_val)
        score_test = _anomaly_score(model, X_test)
        predict_seconds = time.perf_counter() - t1

        train_pr_auc = pr_auc(y_train, score_train)
        val_pr_auc = pr_auc(y_val, score_val)
        test_pr_auc = pr_auc(y_test, score_test)

        threshold_result = best_f1_threshold(y_val, score_val)
        val_f1 = threshold_result.f1
        test_f1 = f1_at_threshold(y_test, score_test, threshold_result.threshold)

        return BenchmarkResult(
            spec=spec,
            train_score=train_pr_auc,
            val_score=val_pr_auc,
            test_score=test_pr_auc,
            val_secondary=val_f1,
            test_secondary=test_f1,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            model_size_bytes=_safe_pickle_size(model),
            threshold=threshold_result.threshold,
            extra={
                "val_precision": threshold_result.precision,
                "val_recall": threshold_result.recall,
            },
        )

    def _evaluate_rul(
        self,
        spec: ModelSpec,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) -> BenchmarkResult:
        model = spec.factory()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        predict_seconds = time.perf_counter() - t1

        train_mae = mae(y_train, train_pred)
        val_mae = mae(y_val, val_pred)
        test_mae = mae(y_test, test_pred)

        return BenchmarkResult(
            spec=spec,
            train_score=-train_mae,
            val_score=-val_mae,
            test_score=-test_mae,
            train_secondary=train_mae,
            val_secondary=val_mae,
            test_secondary=test_mae,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            model_size_bytes=_safe_pickle_size(model),
            threshold=None,
            extra={
                "val_mae_hours": val_mae,
                "test_mae_hours": test_mae,
            },
        )

    def _overfit_flags(self, results: Sequence[BenchmarkResult]) -> list[OverfitFlag]:
        flags = []
        for r in results:
            if r.error is not None:
                continue
            gap = abs(r.val_score - r.test_score)
            if gap > self.overfit_tolerance:
                flags.append(
                    OverfitFlag(
                        name=r.spec.name,
                        val_score=r.val_score,
                        test_score=r.test_score,
                        gap=gap,
                        tolerance=self.overfit_tolerance,
                    )
                )
        return flags

    def _select_winner(
        self,
        results: Sequence[BenchmarkResult],
        overfit_flags: Sequence[OverfitFlag],
    ) -> BenchmarkResult | None:
        flagged = {flag.name for flag in overfit_flags}
        candidates = [r for r in results if r.error is None and r.spec.name not in flagged]
        if not candidates:
            return None
        candidates.sort(
            key=lambda r: (-r.val_score, -r.test_score, r.spec.name),
        )
        return candidates[0]


def _anomaly_score(model, X) -> np.ndarray:
    """Convert an arbitrary sklearn detector's output to "higher = anomaly"."""

    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(X), dtype=np.float64)
    if hasattr(model, "decision_function"):
        return -np.asarray(model.decision_function(X), dtype=np.float64)
    if hasattr(model, "predict"):
        preds = np.asarray(model.predict(X), dtype=np.float64)
        return (preds == -1).astype(np.float64)
    raise AttributeError(
        f"model {type(model).__name__} has no score_samples / decision_function / predict"
    )


def default_anomaly_models() -> list[ModelSpec]:
    """The five-model anomaly panel the Phase 2A brief mandates.

    Each spec carries the literature reference for its
    ``contamination`` / kernel choice; nothing here is hand-tuned. The
    same five names appear in the comparison report.
    """

    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    contamination = 0.05
    return [
        ModelSpec(
            name="IsolationForest",
            factory=lambda: IsolationForest(
                contamination=contamination, n_estimators=100, random_state=42
            ),
            family="anomaly",
            description=(
                "Tree-based ensemble; isolation paths length scores "
                "outliers (Liu, Ting & Zhou 2008). Baseline detector."
            ),
            hyperparameters={
                "contamination": contamination,
                "n_estimators": 100,
                "random_state": 42,
            },
        ),
        ModelSpec(
            name="LocalOutlierFactor",
            factory=lambda: LocalOutlierFactor(
                n_neighbors=20, contamination=contamination, novelty=True
            ),
            family="anomaly",
            description=(
                "Density-ratio score against k-nearest neighbours "
                "(Breunig et al. 2000). Non-tree alternative."
            ),
            hyperparameters={
                "n_neighbors": 20,
                "contamination": contamination,
                "novelty": True,
            },
        ),
        ModelSpec(
            name="OneClassSVM",
            factory=lambda: OneClassSVM(nu=contamination, kernel="rbf", gamma="scale"),
            family="anomaly",
            description=(
                "Maximum-margin support estimation in feature space "
                "(Schölkopf et al. 2001). Kernel alternative."
            ),
            hyperparameters={"nu": contamination, "kernel": "rbf", "gamma": "scale"},
        ),
        ModelSpec(
            name="EllipticEnvelope",
            factory=lambda: EllipticEnvelope(
                contamination=contamination, random_state=42, support_fraction=None
            ),
            family="anomaly",
            description=(
                "Minimum Covariance Determinant fit assuming a Gaussian "
                "core (Rousseeuw & Van Driessen 1999)."
            ),
            hyperparameters={
                "contamination": contamination,
                "support_fraction": None,
                "random_state": 42,
            },
        ),
        ModelSpec(
            name="AutoEncoderMLP",
            factory=_autoencoder_factory,
            family="anomaly",
            description=(
                "MLP auto-encoder reconstruction error (Hinton & "
                "Salakhutdinov 2006). Neural alternative built on the "
                "existing scikit-learn stack."
            ),
            hyperparameters={
                "hidden_layer_sizes": (16, 4, 16),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 200,
                "random_state": 42,
            },
        ),
    ]


def _autoencoder_factory():
    """Build an MLP-based auto-encoder anomaly scorer (sklearn only).

    The detector trains a ``MLPRegressor`` to reconstruct ``X`` from
    ``X``; per-sample anomaly score is the L2 reconstruction error. A
    thin wrapper exposes ``fit(X)`` / ``score_samples(X)`` so it slots
    into the same evaluation loop as the other detectors.
    """

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    class _AutoEncoder:
        def __init__(self):
            self.scaler = StandardScaler()
            self.network = MLPRegressor(
                hidden_layer_sizes=(16, 4, 16),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=42,
            )

        def fit(self, X):
            X_scaled = self.scaler.fit_transform(X)
            self.network.fit(X_scaled, X_scaled)
            return self

        def score_samples(self, X):
            X_scaled = self.scaler.transform(X)
            recon = self.network.predict(X_scaled)
            err = np.mean((X_scaled - recon) ** 2, axis=1)
            return -err

    return _AutoEncoder()


def default_rul_models() -> list[ModelSpec]:
    """The four-model RUL panel the Phase 2A brief mandates."""

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None

    specs: list[ModelSpec] = []
    if XGBRegressor is not None:
        specs.append(
            ModelSpec(
                name="XGBoost",
                factory=lambda: XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                    n_jobs=2,
                ),
                family="rul",
                description=(
                    "Gradient-boosted regression trees (Chen & Guestrin "
                    "2016); matches the existing RULPredictor baseline."
                ),
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "random_state": 42,
                },
                needs_fit_y=True,
            )
        )
    specs.extend(
        [
            ModelSpec(
                name="RandomForest",
                factory=lambda: RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=2,
                    random_state=42,
                ),
                family="rul",
                description=(
                    "Bagged regression trees (Breiman 2001). Strong "
                    "non-boosting baseline; supports out-of-bag variance."
                ),
                hyperparameters={
                    "n_estimators": 200,
                    "min_samples_leaf": 2,
                    "max_depth": None,
                    "random_state": 42,
                },
                needs_fit_y=True,
            ),
            ModelSpec(
                name="GradientBoosting",
                factory=lambda: GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                ),
                family="rul",
                description=(
                    "Sklearn-native gradient boosting (Friedman 2001). "
                    "Pure-Python stand-in when XGBoost is unavailable."
                ),
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "random_state": 42,
                },
                needs_fit_y=True,
            ),
            ModelSpec(
                name="WeibullOnly",
                factory=_weibull_factory,
                family="rul",
                description=(
                    "Parametric baseline: fits a 2-parameter Weibull "
                    "(Weibull 1951) to observed RUL labels and returns "
                    "the conditional median. No feature use."
                ),
                hyperparameters={"distribution": "weibull_min", "shape_floor": 0.5},
                needs_fit_y=True,
            ),
        ]
    )
    return specs


def _weibull_factory():
    """Construct a parametric Weibull RUL baseline matching the brief."""

    import scipy.stats as stats

    class _Weibull:
        def __init__(self):
            self.shape = None
            self.scale = None
            self.median = None

        def fit(self, X, y):
            y = np.asarray(y, dtype=np.float64)
            if np.any(y < 0):
                raise ValueError("Weibull baseline requires non-negative RUL labels")
            shape, loc, scale = stats.weibull_min.fit(y, floc=0.0)
            self.shape = max(0.5, float(shape))
            self.scale = max(1.0, float(scale))
            self.median = float(stats.weibull_min.median(self.shape, loc=loc, scale=self.scale))
            return self

        def predict(self, X):
            n = np.asarray(X).shape[0]
            return np.full(n, self.median, dtype=np.float64)

    return _Weibull()


__all__ = [
    "BenchmarkReport",
    "BenchmarkResult",
    "DEFAULT_OVERFIT_TOLERANCE",
    "ModelBenchmark",
    "ModelSpec",
    "OverfitFlag",
    "default_anomaly_models",
    "default_rul_models",
]
