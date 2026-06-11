"""
src/ml/evaluation/optuna_search.py
===========================================
Phase 6.2 cherry-pick: Optuna-driven hyperparameter search for the
existing model zoo.

Background
----------
The Phase 2A ``ModelBenchmark`` runs each ``ModelSpec`` with static
hyperparameters drawn from the cited literature defaults. The
``PHASE_6B_DS_AUDIT.md`` flagged this as the single biggest
methodological gap (Dimension 5, "no hyperparameter tuning"). This
module closes the gap with a thin wrapper over ``optuna`` that:

* defines a documented search space for each of the four model
  families the production stack cares about — ``GradientBoosting``,
  ``XGBoost``, ``LightGBM`` (if available), and ``IsolationForest``;
* uses TPE (Tree-structured Parzen Estimator) sampling with a
  ``MedianPruner`` to skip poor trials early;
* evaluates every trial against the existing
  :class:`~src.ml.evaluation.temporal_split.PurgedTimeSeriesCV`
  so the same leak-free embargo honoured by ``ModelBenchmark`` is
  applied here — no per-machine leakage is introduced;
* persists the best parameters to a JSON artifact next to the
  model card so the production training path can pick them up;
* saves an Optuna HTML visualization per study for the technical
  reviewer (``plot_parallel_coordinate`` + ``plot_param_importances``).

Cherry-pick provenance
----------------------
The pattern is adapted from ``pdm-v3/src/ml/models.py:208-302`` with
three deliberate changes for the Kopya codebase:

1. The CV folds come from ``PurgedTimeSeriesCV`` (already in
   ``src/ml/evaluation/temporal_split.py``), not from
   ``pdm-v3``'s hand-rolled ``_build_machine_aware_cv``. This is the
   same splitter the existing benchmark uses, so a tuned model
   chosen here is comparable to a literature-default model chosen
   by ``ModelBenchmark``.

2. The objective metric is configurable (``pr_auc`` for anomaly,
   ``-mae`` for RUL) so the same module handles both families.

3. The model registry is local to this file (no global mutable
   ``MODEL_REGISTRY`` at import time — optuna is a heavyweight
   import, deferred to first use).

References
----------
* Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter
  Optimization Framework." KDD.
* Bergstra & Bengio (2012). "Random Search for Hyper-Parameter
  Optimization." JMLR 13:281-305 (the TPE motivation).
* López de Prado (2018). "Advances in Financial Machine Learning."
  Wiley, ch. 7 (purged k-fold + embargo).
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy optuna import — optuna is heavy (numpy, scipy, alembic, ...) and
# some test contexts may not have it installed. We import inside the
# functions that need it so the import-time cost is paid only on use.
_optuna = None


def _get_optuna():
    global _optuna
    if _optuna is None:
        import optuna

        _optuna = optuna
    return _optuna


# ────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class OptunaSearchResult:
    """Outcome of one Optuna study."""

    model_name: str
    family: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    metric: str
    artifact_dir: str | None = None
    best_params_path: str | None = None
    study_html_path: str | None = None
    study_pickle_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary suitable for a model card."""
        out: dict[str, Any] = {
            "model_name": self.model_name,
            "family": self.family,
            "metric": self.metric,
            "best_value": float(self.best_value),
            "n_trials": int(self.n_trials),
            "best_params": dict(self.best_params),
        }
        if self.best_params_path:
            out["best_params_path"] = self.best_params_path
        if self.study_html_path:
            out["study_html_path"] = self.study_html_path
        return out


# ────────────────────────────────────────────────────────────────────────────
# Search spaces (Optuna ``suggest_*`` callables)
# ────────────────────────────────────────────────────────────────────────────


def _gb_params(trial: Any) -> dict[str, Any]:
    """GradientBoostingRegressor search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": trial.suggest_categorical("random_state", [42]),
    }


def _xgb_params(trial: Any) -> dict[str, Any]:
    """XGBoost regressor search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": 2,
    }


def _lgbm_params(trial: Any) -> dict[str, Any]:
    """LightGBM regressor search space (skipped if lightgbm is missing)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": 2,
        "verbosity": -1,
    }


def _if_params(trial: Any) -> dict[str, Any]:
    """IsolationForest search space (anomaly detection)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_samples": trial.suggest_categorical(
            "max_samples", ["auto", 256, 512, 1024]
        ),
        "contamination": trial.suggest_float("contamination", 0.01, 0.20),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "random_state": 42,
    }


# Registry keyed by the same name used in ``default_rul_models()`` /
# ``default_anomaly_models()`` so the tuned hyperparameters slot
# directly into a ``ModelSpec.factory`` with no further translation.
SEARCH_SPACES: dict[str, Callable[[Any], dict[str, Any]]] = {
    "GradientBoosting": _gb_params,
    "XGBoost": _xgb_params,
    "IsolationForest": _if_params,
}

# LightGBM is optional — only registered if the package is importable.
try:
    import lightgbm  # noqa: F401

    SEARCH_SPACES["LightGBM"] = _lgbm_params
except ImportError:
    pass


# ────────────────────────────────────────────────────────────────────────────
# Family → metric and score-fn mapping
# ────────────────────────────────────────────────────────────────────────────


def _make_objective(
    *,
    model_name: str,
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    machine_ids: np.ndarray | None,
    splitter_factory: Callable[[], Any],
    random_state: int,
    n_folds: int,
    metric_name: str,
):
    """Build the Optuna objective for a model.

    The objective scores the model on each CV fold and returns the
    mean of the metric. For ``rul`` the metric is ``-mae`` (higher
    is better, so Optuna maximises); for ``anomaly`` the metric is
    ``pr_auc`` (already higher-is-better).
    """
    from src.ml.evaluation.metrics import mae, pr_auc

    optuna = _get_optuna()

    if family == "rul":
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from xgboost import XGBRegressor

        factories: dict[str, Any] = {
            "XGBoost": lambda: XGBRegressor(),
            "GradientBoosting": lambda: GradientBoostingRegressor(),
            "RandomForest": lambda: RandomForestRegressor(),
        }
        if "LightGBM" in SEARCH_SPACES:
            import lightgbm as lgb

            factories["LightGBM"] = lambda: lgb.LGBMRegressor()
    elif family == "anomaly":
        from sklearn.ensemble import IsolationForest

        factories = {"IsolationForest": lambda: IsolationForest()}
    else:
        raise ValueError(f"family must be 'rul' or 'anomaly', got {family!r}")

    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {list(SEARCH_SPACES)}"
        )
    if model_name not in factories:
        raise ValueError(
            f"Model {model_name!r} has a search space but no factory for family={family!r}"
        )
    param_fn = SEARCH_SPACES[model_name]
    factory = factories[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = param_fn(trial)
        if "random_state" in params and "random_state" not in params:
            params["random_state"] = random_state
        trial.set_user_attr("full_params", dict(params))

        cv = splitter_factory()
        fold_scores: list[float] = []
        for fold in cv.split(timestamps, groups=machine_ids):
            train_idx = fold.train_idx
            val_idx = fold.val_idx
            if train_idx.size == 0 or val_idx.size == 0:
                continue
            try:
                model = factory()
                model.set_params(**params)
                if family == "anomaly":
                    # Unsupervised — labels are used only for scoring.
                    model.fit(X[train_idx])
                    if hasattr(model, "score_samples"):
                        scores = -np.asarray(model.score_samples(X[val_idx]))
                    elif hasattr(model, "decision_function"):
                        scores = -np.asarray(model.decision_function(X[val_idx]))
                    else:
                        preds = np.asarray(model.predict(X[val_idx]))
                        scores = (preds == -1).astype(np.float64)
                    score = float(pr_auc(y[val_idx], scores))
                else:
                    model.fit(X[train_idx], y[train_idx])
                    preds = model.predict(X[val_idx])
                    # Maximise -MAE → higher is better.
                    score = -float(mae(y[val_idx], preds))
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Optuna trial failed for %s with params=%s: %s",
                    model_name,
                    params,
                    exc,
                )
                # 0.0 = worst plausible score; the pruner will skip
                # this trial on subsequent folds.
                return 0.0
            fold_scores.append(score)

        if not fold_scores:
            return 0.0
        mean_score = float(np.mean(fold_scores))
        return mean_score

    return objective


# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────


def _resolve_factories():
    """Lazy import of model factories to keep this module's import cheap."""
    from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
    from xgboost import XGBRegressor

    out: dict[str, Any] = {
        "XGBoost": XGBRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "IsolationForest": IsolationForest,
        "RandomForest": RandomForestRegressor,
    }
    try:
        import lightgbm as lgb

        out["LightGBM"] = lgb.LGBMRegressor
    except ImportError:
        pass
    return out


def _make_artifact_paths(artifact_dir: str | None, model_name: str) -> dict[str, str]:
    if artifact_dir is None:
        artifact_dir = os.path.join("model_store", f"{model_name}_tuning")
    os.makedirs(artifact_dir, exist_ok=True)
    return {
        "artifact_dir": artifact_dir,
        "best_params_path": os.path.join(artifact_dir, "best_params.json"),
        "study_html_path": os.path.join(artifact_dir, "study.html"),
        "study_pickle_path": os.path.join(artifact_dir, "study.pkl"),
    }


def tune_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    *,
    family: str = "rul",
    machine_ids: np.ndarray | None = None,
    n_splits: int = 3,
    embargo_seconds: float = 300.0,
    n_trials: int = 50,
    random_state: int = 42,
    metric: str | None = None,
    artifact_dir: str | None = None,
    verbose: bool = False,
    seed: int | None = None,
) -> OptunaSearchResult:
    """Run an Optuna study for one model on the leak-free purged CV.

    Parameters
    ----------
    model_name
        Key in :data:`SEARCH_SPACES` (e.g. ``"XGBoost"``).
    X, y
        Feature matrix and target vector.
    timestamps
        Per-row timestamp array (any dtype coercible to
        ``np.datetime64[ns]``). Required because the splitter is
        :class:`PurgedTimeSeriesCV`; we cannot bypass it.
    family
        ``"rul"`` (regression; ``-mae`` is maximised) or
        ``"anomaly"`` (unsupervised; ``pr_auc`` of the held-out
        fold is maximised).
    machine_ids
        Optional per-row machine identifier so the splitter can keep
        each machine's chronology intact. ``None`` treats the whole
        dataset as one virtual machine.
    n_splits
        Number of purged k-fold splits.
    embargo_seconds
        Embargo on each side of every validation window.
    n_trials
        Optuna trial budget. Production default is 50; tests use 3-5.
    random_state
        TPE sampler seed.
    metric
        Optional override; default is ``"-mae"`` for ``rul`` and
        ``"pr_auc"`` for ``anomaly``.
    artifact_dir
        Where to write ``best_params.json`` + ``study.html`` (default
        ``model_store/<model>_tuning/``).
    verbose
        If ``False`` (default), suppress Optuna's progress logs.
    seed
        Deprecated alias for ``random_state``. Kept for backward
        compatibility with the v3 call signature; if both are given,
        ``random_state`` wins.

    Returns
    -------
    OptunaSearchResult
        Dataclass with the best parameters, best value, and the
        filesystem paths to the artifacts.
    """
    if seed is not None and random_state == 42:
        random_state = int(seed)

    optuna = _get_optuna()

    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"Unknown model {model_name!r}. Available: {list(SEARCH_SPACES)}"
        )
    if family not in ("rul", "anomaly"):
        raise ValueError(f"family must be 'rul' or 'anomaly', got {family!r}")
    if metric is None:
        metric = "pr_auc" if family == "anomaly" else "-mae"

    paths = _make_artifact_paths(artifact_dir, model_name)

    def _splitter_factory():
        from src.ml.evaluation.temporal_split import PurgedTimeSeriesCV

        return PurgedTimeSeriesCV(
            n_splits=n_splits,
            embargo_seconds=embargo_seconds,
        )

    objective = _make_objective(
        model_name=model_name,
        family=family,
        X=np.asarray(X, dtype=np.float64),
        y=np.asarray(y),
        timestamps=np.asarray(timestamps),
        machine_ids=machine_ids,
        splitter_factory=_splitter_factory,
        random_state=random_state,
        n_folds=n_splits,
        metric_name=metric,
    )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(2, n_trials // 10),
        n_warmup_steps=0,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name}_tune_{family}",
    )

    log_level = optuna.logging.INFO if verbose else optuna.logging.WARNING
    optuna.logging.set_verbosity(log_level)

    sink = io.StringIO() if not verbose else None
    if sink is not None:
        with redirect_stdout(sink), redirect_stderr(sink):
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    # Merge in any non-searched constants (e.g. random_state, n_jobs)
    # that the objective recorded on the winning trial.
    best_full = study.best_trial.user_attrs.get("full_params") or best_params

    best_params_payload = {
        "model_name": model_name,
        "family": family,
        "metric": metric,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "embargo_seconds": embargo_seconds,
        "random_state": random_state,
        "best_value": float(study.best_value),
        "best_params": {k: _jsonable(v) for k, v in best_full.items()},
        "searched_params": {k: _jsonable(v) for k, v in best_params.items()},
        "tuned_at": datetime.now(UTC).isoformat(),
    }
    with open(paths["best_params_path"], "w", encoding="utf-8") as f:
        json.dump(best_params_payload, f, indent=2, sort_keys=True)

    # HTML visualizations — saved but never raised if the rendering
    # backend (matplotlib) is not available.
    html_path: str | None = None
    try:
        import optuna.visualization as vis

        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_importance = vis.plot_param_importances(study)
        try:
            from plotly.io import to_html

            html_blob = (
                "<html><head><title>Optuna study — "
                f"{model_name}</title></head><body>"
                "<h1>Parallel coordinate</h1>"
                + to_html(fig_parallel, include_plotlyjs="cdn", full_html=False)
                + "<h1>Parameter importances</h1>"
                + to_html(fig_importance, include_plotlyjs="cdn", full_html=False)
                + "</body></html>"
            )
            with open(paths["study_html_path"], "w", encoding="utf-8") as f:
                f.write(html_blob)
            html_path = paths["study_html_path"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Optuna HTML export skipped: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Optuna visualization backend unavailable: %s", exc)

    # Persist the raw study for downstream analysis.
    try:
        import pickle

        with open(paths["study_pickle_path"], "wb") as f:
            pickle.dump(study, f)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Study pickling failed: %s", exc)

    return OptunaSearchResult(
        model_name=model_name,
        family=family,
        best_params={k: _jsonable(v) for k, v in best_full.items()},
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        metric=metric,
        artifact_dir=paths["artifact_dir"],
        best_params_path=paths["best_params_path"],
        study_html_path=html_path,
        study_pickle_path=paths.get("study_pickle_path"),
        extra={
            "n_splits": n_splits,
            "embargo_seconds": embargo_seconds,
            "random_state": random_state,
        },
    )


def tune_all(
    model_names: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    *,
    family: str = "rul",
    machine_ids: np.ndarray | None = None,
    n_splits: int = 3,
    embargo_seconds: float = 300.0,
    n_trials: int = 50,
    random_state: int = 42,
    artifact_root: str | None = None,
    verbose: bool = False,
) -> list[OptunaSearchResult]:
    """Run :func:`tune_model` for every name in ``model_names``.

    Results are returned sorted by ``best_value`` (descending) — the
    first element is the best model. Each study is written to its
    own subdirectory under ``artifact_root`` (default
    ``model_store/``).
    """
    if artifact_root is None:
        artifact_root = "model_store"

    results: list[OptunaSearchResult] = []
    for name in model_names:
        out_dir = os.path.join(artifact_root, f"{name}_tuning")
        results.append(
            tune_model(
                name,
                X,
                y,
                timestamps,
                family=family,
                machine_ids=machine_ids,
                n_splits=n_splits,
                embargo_seconds=embargo_seconds,
                n_trials=n_trials,
                random_state=random_state,
                artifact_dir=out_dir,
                verbose=verbose,
            )
        )

    # Higher metric is better; sort descending.
    results.sort(key=lambda r: r.best_value, reverse=True)
    return results


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def _jsonable(v: Any) -> Any:
    """Coerce numpy / sklearn types to JSON-serialisable Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def main(argv: Sequence[str | None] = None) -> int:
    """CLI entry point.

    Example::

        python -m src.ml.evaluation.optuna_search \
            --model XGBoost --family rul --n-trials 100 \
            --output model_store/xgb_tuning/
    """
    import argparse

    import polars as pl

    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES) or ["XGBoost"])
    parser.add_argument("--family", default="rul", choices=["rul", "anomaly"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--embargo-seconds", type=float, default=300.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a parquet/csv with columns timestamp, machine_id, value, label",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for the best_params.json + study.html",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.data is not None:
        # Load a real dataset from disk; otherwise synthesise a small
        # toy dataset so the CLI is usable in a smoke test.
        if args.data.endswith(".parquet"):
            df = pl.read_parquet(args.data)
        else:
            df = pl.read_csv(args.data)
        ts = df["timestamp"].to_numpy()
        mid = df["machine_id"].to_numpy() if "machine_id" in df.columns else None
        feature_cols = [c for c in df.columns if c not in ("timestamp", "machine_id", "label")]
        X = df.select(feature_cols).fill_null(0.0).to_numpy()
        y = df["label"].to_numpy() if "label" in df.columns else np.zeros(len(df), dtype=np.int64)
    else:
        rng = np.random.default_rng(args.random_state)
        n = 600
        ts0 = np.datetime64("2026-01-01T00:00:00", "ns")
        ts = ts0 + np.arange(n, dtype=np.int64) * np.timedelta64(10, "s")
        # Synthesise 2 "machines" so the purged CV produces ≥3 folds
        # on a 600-row fixture (the splitter needs n_splits+1 per machine).
        mid = np.array([f"M-{i // 300:02d}" for i in range(n)], dtype=object)
        X = rng.normal(0, 1, size=(n, 4))
        y = (X[:, 0] + 0.5 * rng.normal(0, 1, size=n) > 0).astype(np.int64) if args.family == "anomaly" else X @ np.array([1.0, 0.5, -0.3, 0.2]) + rng.normal(0, 0.1, size=n)

    result = tune_model(
        args.model,
        X,
        y,
        ts,
        family=args.family,
        machine_ids=mid,
        n_splits=args.n_splits,
        embargo_seconds=args.embargo_seconds,
        n_trials=args.n_trials,
        random_state=args.random_state,
        artifact_dir=args.output,
        verbose=args.verbose,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
