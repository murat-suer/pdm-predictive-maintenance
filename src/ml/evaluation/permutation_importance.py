"""
src/ml/evaluation/permutation_importance.py
====================================================
Phase 6.3 cherry-pick: Permutation feature importance for the
production model zoo.

Background
----------
The Phase 2A benchmark reports ``val_score`` and ``test_score`` but
not *which features drove the prediction*. Without that, a peer
reviewer cannot tell whether the model is exploiting the right
sensor channels. SHAP is already wired into the anomaly detector
(``AnomalyDetector._compute_shap``) but is heavy (50-200ms per row)
and tree-specific. Permutation importance is the standard
complement: model-agnostic, fast, and the right diagnostic to show
on the model card.

References
----------
* Breiman (2001). "Random Forests." Machine Learning 45(1).
  §10 — the original permutation importance definition.
* Altmann et al. (2010). "Permutation importance: a corrected
  feature importance measure." Bioinformatics 26(10):1340-1347.
* scikit-learn ``inspection.permutation_importance`` — the
  implementation we wrap (no new dependencies).

Cherry-pick provenance
----------------------
Adapted from ``pdm-v3/src/ml/pipeline.py:485-523`` with one
Kopya-specific change: we save the top-N features to a JSON
artifact and emit a matplotlib horizontal bar chart, so the model
card can surface "Top 10 features driving RUL predictions" without
the dashboard having to call SHAP live.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PermutationImportanceResult:
    """Outcome of one permutation importance study."""

    model_name: str
    family: str
    feature_names: list[str]
    importances_mean: list[float]
    importances_std: list[float]
    n_repeats: int
    scoring: str
    artifact_dir: str | None = None
    json_path: str | None = None
    chart_path: str | None = None
    top_features: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "family": self.family,
            "n_features": len(self.feature_names),
            "n_repeats": self.n_repeats,
            "scoring": self.scoring,
            "top_features": list(self.top_features),
            "importances_mean": [float(x) for x in self.importances_mean],
            "importances_std": [float(x) for x in self.importances_std],
            "feature_names": list(self.feature_names),
            "json_path": self.json_path,
            "chart_path": self.chart_path,
        }


def _resolve_model(model_name: str, family: str):
    """Construct a fresh estimator matching the production model."""
    if family == "rul":
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from xgboost import XGBRegressor

        out: dict[str, Any] = {
            "XGBoost": XGBRegressor,
            "GradientBoosting": GradientBoostingRegressor,
            "RandomForest": RandomForestRegressor,
        }
        try:
            import lightgbm as lgb

            out["LightGBM"] = lgb.LGBMRegressor
        except ImportError:
            pass
    elif family == "anomaly":
        from sklearn.ensemble import IsolationForest

        out = {"IsolationForest": IsolationForest}
    else:
        raise ValueError(f"family must be 'rul' or 'anomaly', got {family!r}")

    if model_name not in out:
        raise ValueError(
            f"Unknown model {model_name!r} for family={family!r}. Available: {list(out)}"
        )
    return out[model_name]


def _scoring_fn(family: str, scoring: str):
    """Return a sklearn-compatible scoring callable."""
    from sklearn.metrics import make_scorer

    from src.ml.evaluation.metrics import mae

    if scoring == "pr_auc":
        return make_scorer(_anomaly_pr_auc_scorer, greater_is_better=True)
    if scoring == "neg_mean_absolute_error":
        return scoring  # sklearn built-in
    if scoring == "mae":
        return make_scorer(mae, greater_is_better=False)
    raise ValueError(f"Unknown scoring {scoring!r}")


def _anomaly_pr_auc_scorer(model, X, y):
    """Wrap a sklearn-style scorer for an unsupervised anomaly detector."""
    from src.ml.evaluation.metrics import pr_auc

    if hasattr(model, "score_samples"):
        scores = -np.asarray(model.score_samples(X), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        scores = -np.asarray(model.decision_function(X), dtype=np.float64)
    else:
        preds = np.asarray(model.predict(X))
        scores = (preds == -1).astype(np.float64)
    return float(pr_auc(np.asarray(y), scores))


def permutation_importance(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    family: str = "rul",
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str | None = None,
    top_n: int = 10,
    artifact_dir: str | None = None,
    fit_kwargs: dict[str, Any | None] = None,
    sample_weight: np.ndarray | None = None,
) -> PermutationImportanceResult:
    """
    Compute permutation feature importance for a fitted estimator.

    The estimator is freshly constructed and fit on the provided
    data so the caller does not have to share state. This is
    intentional: a freshly-fit model on the same data the
    importance is measured on is the textbook setup (Breiman
    2001), and it means the function has no side effects on
    existing production models.

    Parameters
    ----------
    model_name
        Key in the per-family registry (``XGBoost``,
        ``GradientBoosting``, ``RandomForest``, ``LightGBM`` for
        rul; ``IsolationForest`` for anomaly).
    X, y
        Feature matrix and target vector. ``y`` is the binary
        label for anomaly (used for PR-AUC scoring) and the
        numeric RUL for rul.
    feature_names
        Per-column feature name. Must match ``X.shape[1]``.
    family
        ``"rul"`` or ``"anomaly"``.
    n_repeats
        Number of permutations per feature (Breiman recommends
        ≥5; default 5 keeps the test under a second).
    random_state
        Seed for the permutation shuffle.
    scoring
        sklearn scoring string. Default: ``"pr_auc"`` for anomaly,
        ``"neg_mean_absolute_error"`` for rul.
    top_n
        Number of top features to capture in the artifact
        (default 10).
    artifact_dir
        Where to write ``permutation_importance.json`` and the
        PNG chart (default ``model_store/<model>_importance/``).
    fit_kwargs
        Extra kwargs forwarded to the model ``.fit()`` call (e.g.
        ``sample_weight``). Pass None to use the model defaults.
    sample_weight
        Convenience for the most common fit kwarg; merged into
        ``fit_kwargs`` if ``fit_kwargs`` does not already supply
        it.

    Returns
    -------
    PermutationImportanceResult
    """
    if scoring is None:
        scoring = "pr_auc" if family == "anomaly" else "neg_mean_absolute_error"
    if artifact_dir is None:
        artifact_dir = os.path.join("model_store", f"{model_name}_importance")
    os.makedirs(artifact_dir, exist_ok=True)

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    if X_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {X_arr.shape[1]} columns but feature_names has {len(feature_names)}"
        )

    model_cls = _resolve_model(model_name, family)
    model = model_cls()
    fit_kwargs_combined: dict[str, Any] = dict(fit_kwargs or {})
    if sample_weight is not None and "sample_weight" not in fit_kwargs_combined:
        fit_kwargs_combined["sample_weight"] = np.asarray(sample_weight)
    if family == "anomaly":
        # Unsupervised: fit on X only.
        model.fit(X_arr)
    else:
        model.fit(X_arr, y_arr, **fit_kwargs_combined)

    from sklearn.inspection import permutation_importance as _sk_perm

    perm = _sk_perm(
        model,
        X_arr,
        y_arr,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        scoring=scoring,
    )
    means = np.asarray(perm.importances_mean, dtype=np.float64)
    stds = np.asarray(perm.importances_std, dtype=np.float64)

    # Rank by mean importance (descending). Some features can have
    # negative mean importance — they hurt the model when perturbed.
    order = np.argsort(-means)
    top_idx = order[: int(top_n)]
    top_features: list[dict[str, Any]] = [
        {
            "rank": int(rank + 1),
            "feature": str(feature_names[i]),
            "importance_mean": float(means[i]),
            "importance_std": float(stds[i]),
        }
        for rank, i in enumerate(top_idx)
    ]

    json_path = os.path.join(artifact_dir, "permutation_importance.json")
    payload: dict[str, Any] = {
        "model_name": model_name,
        "family": family,
        "scoring": scoring,
        "n_repeats": int(n_repeats),
        "random_state": int(random_state),
        "n_features": int(X_arr.shape[1]),
        "n_samples": int(X_arr.shape[0]),
        "feature_names": list(feature_names),
        "importances_mean": [float(x) for x in means],
        "importances_std": [float(x) for x in stds],
        "top_features": top_features,
        "tuned_at": datetime.now(UTC).isoformat(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    chart_path: str | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot top_n by absolute value so negative importances are
        # also surfaced (a feature that *hurts* the model is
        # diagnostic and a peer reviewer wants to see it).
        abs_means = np.abs(means)
        abs_order = np.argsort(-abs_means)[: int(top_n)]
        labels = [str(feature_names[i]) for i in abs_order]
        values = [float(means[i]) for i in abs_order]
        errs = [float(stds[i]) for i in abs_order]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * int(top_n))))
        y_pos = np.arange(len(labels))
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
        ax.barh(y_pos, values, xerr=errs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linewidth=0.6)
        ax.set_xlabel(f"Permutation importance ({scoring})", fontsize=10)
        ax.set_title(
            f"Top {int(top_n)} features — {model_name} ({family})",
            fontsize=11,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        fig.tight_layout()
        chart_path = os.path.join(artifact_dir, "permutation_importance.png")
        fig.savefig(chart_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Permutation importance chart skipped: %s", exc)
        chart_path = None

    return PermutationImportanceResult(
        model_name=model_name,
        family=family,
        feature_names=list(feature_names),
        importances_mean=[float(x) for x in means],
        importances_std=[float(x) for x in stds],
        n_repeats=int(n_repeats),
        scoring=scoring,
        artifact_dir=artifact_dir,
        json_path=json_path,
        chart_path=chart_path,
        top_features=top_features,
    )


def main(argv: Sequence[str | None] = None) -> int:
    """CLI: ``python -m src.ml.evaluation.permutation_importance --model XGBoost``."""
    import argparse

    parser = argparse.ArgumentParser(description="Permutation feature importance")
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", default="rul", choices=["rul", "anomaly"])
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--data", default=None, help="Parquet/csv with feature columns + 'label'")
    parser.add_argument("--output", default=None, help="Output directory for the JSON + PNG")
    args = parser.parse_args(argv)

    if args.data is None:
        rng = np.random.default_rng(args.random_state)
        n = 300
        feature_names = ["vibration_rms", "bearing_temp", "motor_current", "outlet_pressure", "oil_pressure"]
        X = rng.normal(0, 1, size=(n, len(feature_names)))
        if args.family == "rul":
            y = X @ np.array([2.0, -1.0, 0.5, 0.3, 0.2]) + rng.normal(0, 0.1, n)
        else:
            y = (X[:, 0] + 0.5 * rng.normal(0, 1, n) > 0).astype(np.int64)
    else:
        import polars as pl

        df = pl.read_parquet(args.data) if args.data.endswith(".parquet") else pl.read_csv(args.data)
        feature_names = [c for c in df.columns if c not in ("timestamp", "machine_id", "label")]
        X = df.select(feature_names).fill_null(0.0).to_numpy()
        y = df["label"].to_numpy() if "label" in df.columns else np.zeros(len(df), dtype=np.int64)

    result = permutation_importance(
        args.model,
        X,
        y,
        feature_names,
        family=args.family,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        top_n=args.top_n,
        artifact_dir=args.output,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
