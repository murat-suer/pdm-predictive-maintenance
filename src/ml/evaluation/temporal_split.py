"""
src/ml/evaluation/temporal_split.py
============================================
Leak-proof temporal cross-validation for the PdM ML pipeline.

This module is the single source of truth for *how* training, validation,
and test data are separated. Every model factory inside
:mod:`src.ml.evaluation.model_comparison` and every leakage probe
in :mod:`src.ml.evaluation.leakage_test` routes through this
splitter. Bypassing it (e.g. with `sklearn.model_selection.train_test_split`)
is a hard violation of the project's "no random splits" rule.

Design
------
* **Three-way temporal split.** Each machine is split chronologically into
  ``train`` (oldest 70%), ``val`` (middle 15%), ``test`` (newest 15%).
  No random shuffling. The split happens *per machine* so that machines
  with different cycle lengths do not poison each other's chronology.

* **Purged k-fold with embargo (López de Prado 2018, ch. 7).** Inside a
  fold, we additionally drop an *embargo* of `embargo_seconds` either
  side of each validation window. The embargo accounts for two
  information-flow channels that would otherwise cause leakage:

    1. Rolling features that overlap the boundary between train and
       validation. Our feature engineering uses a 30-row × 10-second
       window (5 minutes; see ``feature_engineering.WINDOW_SIZE``); the
       default embargo of 300 seconds is precisely the largest rolling
       window so no engineered feature can simultaneously look at train
       and validation rows.
    2. Serial correlation in the underlying sensor signal (a value at
       time ``t`` is correlated with values nearby). Even features that
       do not explicitly window the data inherit this correlation. The
       embargo widens the gap between training and held-out data so
       that the test score reflects out-of-sample performance, not the
       autocorrelation of the signal.

* **Fit-on-train-only contract.** A companion guard
  (:class:`FitOnTrainGuard`) wraps any sklearn-like estimator and raises
  on illegal `.fit()` / `.fit_transform()` calls outside the training
  window. The wrapper is enforced by tests; production code may instead
  use the lighter ``assert_fit_on_train_only`` checker.

References
----------
* López de Prado, M. (2018). *Advances in Financial Machine Learning.*
  Wiley. Chapter 7 ("Cross-Validation in Finance") — the formal source
  for the purged k-fold and embargo construction implemented here.
* Bergmeir & Benítez (2012). "On the use of cross-validation for time
  series predictor evaluation." *Information Sciences* 191:192-213 —
  empirical demonstration that random k-fold over-estimates accuracy on
  temporally-correlated data.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_EMBARGO_SECONDS = 300.0
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15


@dataclass(frozen=True)
class TemporalSplit:
    """Index arrays for one machine's train / val / test partitions."""

    machine_id: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_time_range: tuple[np.datetime64, np.datetime64]
    val_time_range: tuple[np.datetime64, np.datetime64]
    test_time_range: tuple[np.datetime64, np.datetime64]


@dataclass
class FoldIndices:
    """Train / val index arrays for one fold of a purged k-fold CV."""

    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    embargo_seconds: float
    purged_count: int = 0
    boundary_left: np.datetime64 | None = field(default=None)
    boundary_right: np.datetime64 | None = field(default=None)


def _coerce_timestamps(timestamps) -> np.ndarray:
    """Return *timestamps* as a contiguous ``np.datetime64[ns]`` array."""

    arr = np.asarray(timestamps)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]", copy=False)
    if arr.dtype.kind in ("i", "u", "f"):
        return arr.astype("datetime64[ns]", copy=False)
    return np.array([np.datetime64(t, "ns") for t in arr], dtype="datetime64[ns]")


def _embargo_timedelta(embargo_seconds: float) -> np.timedelta64:
    """Convert a float in seconds to a precise ``np.timedelta64`` value."""

    nanoseconds = int(round(float(embargo_seconds) * 1_000_000_000))
    return np.timedelta64(nanoseconds, "ns")


class PurgedTimeSeriesCV:
    """Leak-proof temporal cross-validator with embargo.

    Parameters
    ----------
    n_splits
        Number of purged k-fold splits to yield from :meth:`split`.
        Forwarded to ``sklearn.model_selection.TimeSeriesSplit`` so the
        fold geometry is identical to the standard expanding-window
        scheme; the only difference is the embargo trim we apply on top.
    embargo_seconds
        Embargo width in seconds applied on *both* sides of every
        validation window. Default (300 s) matches the maximum rolling
        feature window in ``feature_engineering.WINDOW_SIZE``.
    train_fraction, val_fraction
        Fractions of each machine's chronological data assigned to
        train and validation respectively. The remainder is held out as
        the test set. Defaults of (0.70, 0.15) give a 70/15/15 split.

    Notes
    -----
    This splitter is deterministic: no random number generator is used
    anywhere. The fold geometry is purely a function of the timestamps
    and the configured parameters, so the same input always produces
    the same splits — required for reproducibility audits.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_seconds: float = DEFAULT_EMBARGO_SECONDS,
        train_fraction: float = DEFAULT_TRAIN_FRACTION,
        val_fraction: float = DEFAULT_VAL_FRACTION,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo_seconds < 0:
            raise ValueError("embargo_seconds must be >= 0")
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if train_fraction + val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction must leave room for a test set"
            )

        self.n_splits = int(n_splits)
        self.embargo_seconds = float(embargo_seconds)
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)

    def split_train_val_test(
        self,
        timestamps,
        groups: Sequence | None = None,
    ) -> dict[str, TemporalSplit]:
        """Three-way chronological split per machine.

        Parameters
        ----------
        timestamps
            Array-like of timestamps (any dtype convertible to
            ``np.datetime64[ns]``), one per row.
        groups
            Per-row machine identifier. If ``None``, the entire dataset
            is treated as a single virtual machine named ``"_"``. The
            returned mapping is keyed by group id.

        Returns
        -------
        dict[str, TemporalSplit]
            One entry per machine. Train indices come from the earliest
            ``train_fraction`` of that machine's timeline, val from the
            next ``val_fraction``, test from the remainder.
        """

        ts = _coerce_timestamps(timestamps)
        if groups is None:
            group_arr = np.array(["_"] * ts.size)
        else:
            group_arr = np.asarray(groups)
        if group_arr.size != ts.size:
            raise ValueError(
                f"groups length {group_arr.size} != timestamps length {ts.size}"
            )

        out: dict[str, TemporalSplit] = {}
        for machine_id in sorted({str(g) for g in group_arr}):
            mask = group_arr == machine_id
            machine_idx = np.where(mask)[0]
            if machine_idx.size < 3:
                raise ValueError(
                    f"machine {machine_id!r} has only {machine_idx.size} rows; need >= 3"
                )
            sort_order = np.argsort(ts[machine_idx], kind="stable")
            ordered = machine_idx[sort_order]
            n = ordered.size
            n_train = max(1, int(round(n * self.train_fraction)))
            n_val = max(1, int(round(n * self.val_fraction)))
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
            train_idx = ordered[:n_train]
            val_idx = ordered[n_train : n_train + n_val]
            test_idx = ordered[n_train + n_val :]
            if val_idx.size == 0 or test_idx.size == 0:
                raise ValueError(
                    f"machine {machine_id!r}: split produced empty val/test "
                    f"(n={n}, n_train={n_train}, n_val={n_val})"
                )
            out[machine_id] = TemporalSplit(
                machine_id=str(machine_id),
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_time_range=(ts[train_idx[0]], ts[train_idx[-1]]),
                val_time_range=(ts[val_idx[0]], ts[val_idx[-1]]),
                test_time_range=(ts[test_idx[0]], ts[test_idx[-1]]),
            )
        return out

    def split(
        self,
        timestamps,
        groups: Sequence | None = None,
    ) -> Iterator[FoldIndices]:
        """Yield purged k-fold (train, val) index pairs.

        The fold geometry comes from
        ``sklearn.model_selection.TimeSeriesSplit`` (expanding window):
        validation window grows from the front while training expands
        with each fold. After sklearn returns the raw indices we apply
        the embargo: any training row whose timestamp falls within
        ``embargo_seconds`` of the validation window's start or end is
        removed.

        Important: when ``groups`` is supplied, folds are computed
        *independently* per machine and then concatenated. This keeps
        each machine's chronology intact while still producing a
        cross-machine training set inside every fold.
        """

        ts = _coerce_timestamps(timestamps)
        if groups is None:
            group_arr = np.array(["_"] * ts.size)
        else:
            group_arr = np.asarray(groups)
        if group_arr.size != ts.size:
            raise ValueError(
                f"groups length {group_arr.size} != timestamps length {ts.size}"
            )

        embargo = _embargo_timedelta(self.embargo_seconds)
        sklearn_splitter = TimeSeriesSplit(n_splits=self.n_splits)
        per_machine_folds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        per_machine_purged: dict[str, list[int]] = {}
        per_machine_bounds: dict[str, list[tuple[np.datetime64, np.datetime64]]] = {}

        for machine_id in sorted({str(g) for g in group_arr}):
            mask = group_arr == machine_id
            machine_idx = np.where(mask)[0]
            if machine_idx.size < self.n_splits + 1:
                raise ValueError(
                    f"machine {machine_id!r}: {machine_idx.size} rows is fewer "
                    f"than n_splits + 1 = {self.n_splits + 1}; cannot build folds"
                )
            sort_order = np.argsort(ts[machine_idx], kind="stable")
            ordered = machine_idx[sort_order]
            ordered_ts = ts[ordered]

            folds: list[tuple[np.ndarray, np.ndarray]] = []
            purged_counts: list[int] = []
            bounds: list[tuple[np.datetime64, np.datetime64]] = []
            for raw_train_pos, val_pos in sklearn_splitter.split(ordered):
                train_global = ordered[raw_train_pos]
                val_global = ordered[val_pos]
                if val_global.size == 0:
                    continue
                val_start = ordered_ts[val_pos[0]]
                val_end = ordered_ts[val_pos[-1]]
                lower = val_start - embargo
                upper = val_end + embargo
                train_ts = ts[train_global]
                keep = (train_ts < lower) | (train_ts > upper)
                kept_train = train_global[keep]
                purged_counts.append(int((~keep).sum()))
                folds.append((kept_train, val_global))
                bounds.append((val_start, val_end))
            per_machine_folds[machine_id] = folds
            per_machine_purged[machine_id] = purged_counts
            per_machine_bounds[machine_id] = bounds

        machine_ids = list(per_machine_folds.keys())
        n_folds = min(len(per_machine_folds[m]) for m in machine_ids) if machine_ids else 0

        for fold_id in range(n_folds):
            train_chunks = []
            val_chunks = []
            purged_total = 0
            earliest_left = None
            latest_right = None
            for machine_id in machine_ids:
                train_idx, val_idx = per_machine_folds[machine_id][fold_id]
                train_chunks.append(train_idx)
                val_chunks.append(val_idx)
                purged_total += per_machine_purged[machine_id][fold_id]
                left, right = per_machine_bounds[machine_id][fold_id]
                if earliest_left is None or left < earliest_left:
                    earliest_left = left
                if latest_right is None or right > latest_right:
                    latest_right = right
            yield FoldIndices(
                fold=fold_id,
                train_idx=np.concatenate(train_chunks) if train_chunks else np.array([], dtype=np.int64),
                val_idx=np.concatenate(val_chunks) if val_chunks else np.array([], dtype=np.int64),
                embargo_seconds=self.embargo_seconds,
                purged_count=purged_total,
                boundary_left=earliest_left,
                boundary_right=latest_right,
            )


class FitOnTrainGuard:
    """Wrap a fit/transform estimator to enforce the fit-on-train-only rule.

    The wrapped estimator tracks whether it has been fit. Subsequent
    `.fit()` or `.fit_transform()` calls raise
    :class:`FitOnValOrTestError`. Use this in tests and in any pipeline
    that processes a sequence of (train, val, test) splits: fit the
    guard on train, then call only `.transform()` on val/test.

    The guard intentionally does NOT silently re-fit. Re-fitting on a
    second "training" call would mask the very leak this class is built
    to prevent.
    """

    def __init__(self, estimator, name: str | None = None):
        if not (hasattr(estimator, "fit") and hasattr(estimator, "transform")):
            raise TypeError(
                f"estimator {type(estimator).__name__} is missing fit/transform"
            )
        self.estimator = estimator
        self.name = name or type(estimator).__name__
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, X, y=None):
        if self._fitted:
            raise FitOnValOrTestError(
                f"{self.name}: .fit() called twice; refusing to refit on a non-train split"
            )
        self.estimator.fit(X, y) if y is not None else self.estimator.fit(X)
        self._fitted = True
        return self

    def fit_transform(self, X, y=None):
        if self._fitted:
            raise FitOnValOrTestError(
                f"{self.name}: .fit_transform() called after fitting — "
                "use .transform() on val/test only"
            )
        result = (
            self.estimator.fit_transform(X, y)
            if y is not None
            else self.estimator.fit_transform(X)
        )
        self._fitted = True
        return result

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}: .transform() called before .fit()"
            )
        return self.estimator.transform(X)


class FitOnValOrTestError(RuntimeError):
    """Raised when a guarded estimator is re-fit on a non-training split."""


def assert_fit_on_train_only(
    estimators: Iterable[FitOnTrainGuard],
) -> None:
    """Validate that every guard has been fit exactly once."""

    for est in estimators:
        if not est.is_fitted:
            raise RuntimeError(
                f"{est.name}: was never fit on training data"
            )


__all__ = [
    "DEFAULT_EMBARGO_SECONDS",
    "DEFAULT_TRAIN_FRACTION",
    "DEFAULT_VAL_FRACTION",
    "FitOnTrainGuard",
    "FitOnValOrTestError",
    "FoldIndices",
    "PurgedTimeSeriesCV",
    "TemporalSplit",
    "assert_fit_on_train_only",
]
