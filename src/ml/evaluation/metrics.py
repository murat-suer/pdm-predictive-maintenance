"""
src/ml/evaluation/metrics.py
=====================================
Evaluation metrics for anomaly detection and RUL regression.

All implementations delegate the heavy lifting to `sklearn.metrics`; this
module only provides the thin, validated wrappers the rest of the
evaluation pipeline imports. Keeping the call surface tight makes it
easy to audit which exact formulas are used.

References
----------
* Davis & Goadrich (2006). "The Relationship Between Precision-Recall
  and ROC Curves." ICML — establishes PR-AUC as the preferred
  imbalanced-class metric (over ROC-AUC) when the positive class is
  rare, as is the case for industrial anomaly detection.
* Vovk, Gammerman & Shafer (2005). "Algorithmic Learning in a Random
  World." — the source for split-conformal coverage guarantees.
* Niculescu-Mizil & Caruana (2005). "Predicting Good Probabilities With
  Supervised Learning." ICML — the reliability-diagram (calibration
  plot) construction used by `calibration_plot_data`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
)


@dataclass(frozen=True)
class ThresholdResult:
    """Outcome of the F1-maximising threshold search on a validation set."""

    threshold: float
    f1: float
    precision: float
    recall: float


def _as_1d_float(arr) -> np.ndarray:
    """Coerce *arr* to a 1-D float64 ndarray, raising on shape mismatch."""

    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("metric input is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("metric input contains NaN or inf")
    return a


def _as_1d_int(arr) -> np.ndarray:
    """Coerce *arr* to a 1-D int ndarray containing only 0/1 values."""

    a = np.asarray(arr).ravel().astype(np.int64, copy=False)
    if a.size == 0:
        raise ValueError("label input is empty")
    if not np.isin(a, (0, 1)).all():
        raise ValueError("labels must be binary 0/1")
    return a


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (a.k.a. average precision).

    Returns the `sklearn.metrics.average_precision_score` value in [0, 1].
    For an all-negative or all-positive ground truth, returns the trivial
    prevalence (NaN is never returned — that would break downstream
    comparisons).

    Parameters
    ----------
    y_true
        Binary labels (0 = negative, 1 = positive).
    y_score
        Continuous anomaly scores (higher = more anomalous).
    """

    y_true_arr = _as_1d_int(y_true)
    y_score_arr = _as_1d_float(y_score)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_arr.shape} vs y_score {y_score_arr.shape}"
        )
    pos = int(y_true_arr.sum())
    if pos == 0:
        return 0.0
    if pos == y_true_arr.size:
        return 1.0
    return float(average_precision_score(y_true_arr, y_score_arr))


def f1_at_threshold(y_true, y_score, threshold: float) -> float:
    """F1 score after thresholding `y_score >= threshold` to a binary mask."""

    y_true_arr = _as_1d_int(y_true)
    y_score_arr = _as_1d_float(y_score)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_arr.shape} vs y_score {y_score_arr.shape}"
        )
    y_pred = (y_score_arr >= float(threshold)).astype(np.int64)
    return float(f1_score(y_true_arr, y_pred, zero_division=0))


def best_f1_threshold(y_true, y_score) -> ThresholdResult:
    """Locate the score threshold that maximises F1 on the *validation* set.

    Uses `precision_recall_curve` to enumerate every threshold implied by
    the scores, then evaluates F1 = 2·P·R / (P+R) at each one.

    Important: this MUST be called on the validation split only. Calling
    it on the test split is itself a leak (threshold selection peeks at
    the test labels). The selected threshold is then frozen and applied
    on test via `f1_at_threshold`.

    Returns
    -------
    ThresholdResult
        Threshold, F1, precision, and recall at the optimum. Threshold
        is `numpy.inf` for the degenerate "predict-all-negative" case
        (which still yields F1 = 0).
    """

    y_true_arr = _as_1d_int(y_true)
    y_score_arr = _as_1d_float(y_score)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_arr.shape} vs y_score {y_score_arr.shape}"
        )
    if int(y_true_arr.sum()) == 0:
        return ThresholdResult(threshold=float("inf"), f1=0.0, precision=0.0, recall=0.0)

    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    if thresholds.size == 0:
        return ThresholdResult(threshold=float("inf"), f1=0.0, precision=0.0, recall=0.0)

    denom = precisions + recalls
    f1s = np.zeros_like(denom)
    mask = denom > 0
    f1s[mask] = 2.0 * precisions[mask] * recalls[mask] / denom[mask]

    best = int(np.argmax(f1s))
    return ThresholdResult(
        threshold=float(thresholds[best]),
        f1=float(f1s[best]),
        precision=float(precisions[best]),
        recall=float(recalls[best]),
    )


def mae(y_true, y_pred) -> float:
    """Mean absolute error (MAE) for RUL regression, in the same units as `y_true`."""

    y_true_arr = _as_1d_float(y_true)
    y_pred_arr = _as_1d_float(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )
    return float(mean_absolute_error(y_true_arr, y_pred_arr))


def conformal_coverage(
    y_true,
    y_pred_lower,
    y_pred_upper,
    alpha: float = 0.10,
) -> dict:
    """Empirical coverage of split-conformal prediction intervals.

    Returns the fraction of `y_true` values falling inside the closed
    interval `[y_pred_lower, y_pred_upper]`. The nominal target is
    `1 - alpha`; a well-calibrated conformal predictor produces a value
    no smaller than `1 - alpha - 1/(n+1)` with high probability
    (Vovk, Gammerman & Shafer 2005, eq. 2.6).

    Returns
    -------
    dict with keys ``coverage``, ``nominal``, ``deviation``, ``n``,
    ``inside_count``, ``average_width``. The caller can use ``deviation``
    (signed, `coverage - nominal`) to flag intervals that are too narrow
    (negative) or unnecessarily wide (positive).
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    y_true_arr = _as_1d_float(y_true)
    y_low_arr = _as_1d_float(y_pred_lower)
    y_high_arr = _as_1d_float(y_pred_upper)
    if not (y_true_arr.shape == y_low_arr.shape == y_high_arr.shape):
        raise ValueError("y_true, y_pred_lower, y_pred_upper must share a shape")
    if np.any(y_low_arr > y_high_arr):
        raise ValueError("y_pred_lower must be <= y_pred_upper element-wise")

    inside = (y_true_arr >= y_low_arr) & (y_true_arr <= y_high_arr)
    inside_count = int(inside.sum())
    n = int(y_true_arr.size)
    coverage = inside_count / n
    nominal = 1.0 - alpha
    return {
        "coverage": float(coverage),
        "nominal": float(nominal),
        "deviation": float(coverage - nominal),
        "n": n,
        "inside_count": inside_count,
        "average_width": float(np.mean(y_high_arr - y_low_arr)),
    }


def calibration_plot_data(y_true, y_score, n_bins: int = 10) -> dict:
    """Reliability-diagram data: bin scores into equal-width buckets and
    compare each bucket's mean predicted score to the bucket's observed
    positive frequency (the empirical calibration curve).

    The construction follows Niculescu-Mizil & Caruana (2005); for the
    binary classification case it is identical to
    `sklearn.calibration.calibration_curve(strategy="uniform")`. Returning
    raw arrays (rather than a plot) keeps this module free of matplotlib
    so it can be safely imported by the production runtime.
    """

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    y_true_arr = _as_1d_int(y_true)
    y_score_arr = _as_1d_float(y_score)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_arr.shape} vs y_score {y_score_arr.shape}"
        )
    score_min = float(np.min(y_score_arr))
    score_max = float(np.max(y_score_arr))
    if score_max <= score_min:
        score_max = score_min + 1.0
    edges = np.linspace(score_min, score_max, n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    indices = np.digitize(y_score_arr, edges, right=False) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    bin_centres = []
    mean_scores = []
    positive_freq = []
    counts = []
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_centres.append(0.5 * (edges[b] + edges[b + 1]))
        mean_scores.append(float(y_score_arr[mask].mean()))
        positive_freq.append(float(y_true_arr[mask].mean()))
        counts.append(count)

    return {
        "bin_centres": np.asarray(bin_centres, dtype=np.float64),
        "mean_predicted_score": np.asarray(mean_scores, dtype=np.float64),
        "empirical_positive_rate": np.asarray(positive_freq, dtype=np.float64),
        "counts": np.asarray(counts, dtype=np.int64),
    }


__all__ = [
    "ThresholdResult",
    "pr_auc",
    "f1_at_threshold",
    "best_f1_threshold",
    "mae",
    "conformal_coverage",
    "calibration_plot_data",
]
