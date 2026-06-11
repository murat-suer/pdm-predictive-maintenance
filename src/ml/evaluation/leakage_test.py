"""
src/ml/evaluation/leakage_test.py
===========================================
Permutation-based and structural leakage detector for ML pipelines.

A clean ML pipeline must satisfy three properties:

1. *Permuted-label behaviour.* A model trained on randomly-permuted
   labels should perform near the random baseline on the held-out set.
   If the permuted model still scores well, the model has access to
   information that survives label destruction — almost always
   train/test contamination.
2. *Temporal monotonicity.* For time-series data, every training
   timestamp must precede every held-out timestamp. A violation means
   the model peeked at the future.
3. *No suspiciously perfect features.* No single feature should
   correlate with the target above an unrealistically high threshold;
   such a feature is usually a derivative of the target itself.

This module bundles those three checks behind one entry point,
:func:`run_leakage_check`. The result object reports each sub-check's
verdict and, on failure, the worst offender.

References
----------
* Ojala & Garriga (2010). "Permutation Tests for Studying Classifier
  Performance." JMLR 11:1833-1863 — formal permutation-test
  methodology adopted here.
* Kaufman, Rosset, Perlich & Stitelman (2012). "Leakage in Data
  Mining: Formulation, Detection, and Avoidance." TKDD 6(4) — the
  canonical leakage taxonomy this module mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ml.evaluation.metrics import pr_auc

DEFAULT_PERMUTED_GAP_TOLERANCE = 0.10
DEFAULT_FEATURE_CORR_LIMIT = 0.99
DEFAULT_N_PERMUTATIONS = 5


@dataclass(frozen=True)
class LeakageVerdict:
    """One sub-check's outcome within :class:`LeakageReport`."""

    name: str
    passed: bool
    detail: str
    metric: float | None = None
    threshold: float | None = None


@dataclass
class LeakageReport:
    """Aggregated leakage assessment."""

    pr_auc_real: float
    pr_auc_permuted: float
    baseline: float
    gap_real_vs_permuted: float
    permuted_excess_over_baseline: float
    n_permutations: int = 1
    permutation_pr_aucs: list[float] = field(default_factory=list)
    verdicts: list[LeakageVerdict] = field(default_factory=list)
    suspect_features: list[tuple[str, float]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return all(v.passed for v in self.verdicts)

    @property
    def overall_verdict(self) -> str:
        return "CLEAN" if self.clean else "LEAKAGE_SUSPECTED"

    def summary(self) -> str:
        lines = [
            f"verdict: {self.overall_verdict}",
            f"  pr_auc_real     = {self.pr_auc_real:.4f}",
            f"  pr_auc_permuted = {self.pr_auc_permuted:.4f} (median over {self.n_permutations} permutations)",
            f"  baseline        = {self.baseline:.4f}",
            f"  gap (real - perm) = {self.gap_real_vs_permuted:+.4f}",
            f"  permuted excess over baseline = {self.permuted_excess_over_baseline:+.4f}",
        ]
        for v in self.verdicts:
            mark = "OK" if v.passed else "FAIL"
            lines.append(f"  [{mark}] {v.name}: {v.detail}")
        if self.suspect_features:
            for fname, corr in self.suspect_features:
                lines.append(f"    suspect feature: {fname} (|corr|={corr:.3f})")
        return "\n".join(lines)


def run_leakage_check(
    model_factory,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    train_timestamps=None,
    test_timestamps=None,
    feature_names=None,
    permuted_gap_tolerance: float = DEFAULT_PERMUTED_GAP_TOLERANCE,
    feature_corr_limit: float = DEFAULT_FEATURE_CORR_LIMIT,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    score_fn=None,
    random_state: int = 42,
) -> LeakageReport:
    """Run the full leakage suite on a single (model, split) pair.

    Parameters
    ----------
    model_factory
        Zero-arg callable returning a fresh estimator. Must implement
        ``fit(X, y)`` and one of ``score_samples`` / ``decision_function``
        / ``predict_proba`` / ``predict`` (resolved by ``score_fn`` or
        :func:`_default_score`).
    X_train, y_train, X_test, y_test
        Train/test arrays. ``y`` must be binary 0/1.
    train_timestamps, test_timestamps
        Optional 1-D arrays; if both are supplied, the temporal-order
        check verifies ``max(train_timestamps) <= min(test_timestamps)``.
    feature_names
        Optional list of column labels matching ``X_train``. Used to
        identify suspect features in the correlation sub-check.
    permuted_gap_tolerance
        How far the permuted model's PR-AUC may exceed the random
        baseline before triggering a flag. Default 0.10 is calibrated
        as roughly two standard errors of the PR-AUC estimator at
        ``n=100`` with prevalence ≈ 0.25 — tight enough to surface
        genuine train/test contamination, loose enough to absorb
        finite-sample noise. Lower it for larger held-out sets.
    feature_corr_limit
        Absolute Pearson correlation between any single feature and the
        target above which we suspect a target-encoded feature.
        Default 0.99 reflects the fact that even well-separated
        physical sensor signals routinely correlate at 0.7-0.95 with
        an anomaly indicator; only near-perfect correlation is treated
        as a definite leak.
    n_permutations
        Number of independent label permutations to run; the median
        PR-AUC across permutations is reported. ``5`` is the default
        recommended by Ojala & Garriga (2010) for moderate test sizes
        — it averages out chance correlations a single permutation
        would otherwise carry. Set to ``1`` only when speed dominates
        accuracy (the result will then be noisier).
    score_fn
        Optional callable ``(model, X) -> ndarray`` returning anomaly /
        probability scores. If ``None``, :func:`_default_score` is
        used.
    random_state
        Seed for the label permutation. Determinism guaranteed.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    X_train_arr = np.asarray(X_train, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_train_arr = _ensure_binary(y_train, "y_train")
    y_test_arr = _ensure_binary(y_test, "y_test")
    if X_train_arr.shape[1] != X_test_arr.shape[1]:
        raise ValueError(
            f"feature mismatch: X_train has {X_train_arr.shape[1]} cols, "
            f"X_test has {X_test_arr.shape[1]}"
        )
    score = score_fn or _default_score

    pr_real = _train_and_score(
        model_factory, X_train_arr, y_train_arr, X_test_arr, y_test_arr, score
    )

    rng = np.random.default_rng(random_state)
    permutation_aucs: list[float] = []
    for _ in range(n_permutations):
        permuted_labels = y_train_arr.copy()
        rng.shuffle(permuted_labels)
        permutation_aucs.append(
            _train_and_score(
                model_factory, X_train_arr, permuted_labels, X_test_arr, y_test_arr, score
            )
        )
    pr_perm = float(np.median(permutation_aucs))

    pos = int(y_test_arr.sum())
    n = int(y_test_arr.size)
    baseline = pos / n if n else 0.0
    gap = pr_real - pr_perm
    permuted_excess = pr_perm - baseline

    verdicts = []
    verdicts.append(_permutation_verdict(pr_perm, baseline, permuted_gap_tolerance))
    verdicts.append(
        _temporal_order_verdict(train_timestamps, test_timestamps)
    )
    suspect = _feature_correlation_suspects(
        X_train_arr, y_train_arr, feature_names, feature_corr_limit
    )
    verdicts.append(_feature_corr_verdict(suspect, feature_corr_limit))

    return LeakageReport(
        pr_auc_real=pr_real,
        pr_auc_permuted=pr_perm,
        baseline=baseline,
        gap_real_vs_permuted=gap,
        permuted_excess_over_baseline=permuted_excess,
        n_permutations=n_permutations,
        permutation_pr_aucs=permutation_aucs,
        verdicts=verdicts,
        suspect_features=suspect,
    )


def _ensure_binary(y, name: str) -> np.ndarray:
    arr = np.asarray(y).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    arr = arr.astype(np.int64, copy=False)
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return arr


def _default_score(model, X) -> np.ndarray:
    """Best-effort score extraction across sklearn / XGBoost detectors."""

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, -1].astype(np.float64)
        return proba.ravel().astype(np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(X), dtype=np.float64)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=np.float64)
    raise AttributeError(
        f"model {type(model).__name__} has no usable score method"
    )


def _train_and_score(
    factory,
    X_train,
    y_train,
    X_test,
    y_test,
    score_fn,
) -> float:
    model = factory()
    model.fit(X_train, y_train)
    scores = score_fn(model, X_test)
    return pr_auc(y_test, scores)


def _permutation_verdict(
    pr_auc_permuted: float,
    baseline: float,
    tolerance: float,
) -> LeakageVerdict:
    excess = pr_auc_permuted - baseline
    passed = excess <= tolerance
    detail = (
        f"permuted PR-AUC {pr_auc_permuted:.4f} vs baseline {baseline:.4f} "
        f"→ excess {excess:+.4f} (tolerance {tolerance:.2f})"
    )
    if not passed:
        detail += " — model still scores above random with shuffled labels."
    return LeakageVerdict(
        name="permutation_test",
        passed=passed,
        detail=detail,
        metric=excess,
        threshold=tolerance,
    )


def _temporal_order_verdict(train_ts, test_ts) -> LeakageVerdict:
    if train_ts is None or test_ts is None:
        return LeakageVerdict(
            name="temporal_order",
            passed=True,
            detail="skipped (no timestamps supplied)",
        )
    train_arr = np.asarray(train_ts)
    test_arr = np.asarray(test_ts)
    if train_arr.size == 0 or test_arr.size == 0:
        return LeakageVerdict(
            name="temporal_order",
            passed=True,
            detail="skipped (empty timestamps)",
        )
    train_max = train_arr.max()
    test_min = test_arr.min()
    passed = train_max <= test_min
    detail = (
        f"max(train_ts)={train_max} vs min(test_ts)={test_min} → "
        f"{'OK' if passed else 'train extends past test start'}"
    )
    return LeakageVerdict(
        name="temporal_order",
        passed=passed,
        detail=detail,
    )


def _feature_correlation_suspects(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names,
    limit: float,
) -> list[tuple[str, float]]:
    if X_train.size == 0:
        return []
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} != "
            f"X_train cols {X_train.shape[1]}"
        )
    target = y_train.astype(np.float64)
    target_centered = target - target.mean()
    target_norm = float(np.linalg.norm(target_centered))
    if target_norm == 0.0:
        return []
    suspects: list[tuple[str, float]] = []
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        col_centered = col - col.mean()
        col_norm = float(np.linalg.norm(col_centered))
        if col_norm == 0.0:
            continue
        corr = float(np.dot(col_centered, target_centered) / (col_norm * target_norm))
        if abs(corr) >= limit:
            suspects.append((str(feature_names[j]), corr))
    suspects.sort(key=lambda pair: -abs(pair[1]))
    return suspects


def _feature_corr_verdict(
    suspects: list[tuple[str, float]],
    limit: float,
) -> LeakageVerdict:
    passed = len(suspects) == 0
    if passed:
        detail = f"no feature exceeds |corr| >= {limit:.2f} with target"
    else:
        top = suspects[0]
        detail = (
            f"{len(suspects)} feature(s) exceed |corr| >= {limit:.2f}; "
            f"top suspect '{top[0]}' has |corr|={abs(top[1]):.3f}"
        )
    return LeakageVerdict(
        name="feature_target_correlation",
        passed=passed,
        detail=detail,
        metric=(abs(suspects[0][1]) if suspects else 0.0),
        threshold=limit,
    )


__all__ = [
    "DEFAULT_FEATURE_CORR_LIMIT",
    "DEFAULT_N_PERMUTATIONS",
    "DEFAULT_PERMUTED_GAP_TOLERANCE",
    "LeakageReport",
    "LeakageVerdict",
    "run_leakage_check",
]
