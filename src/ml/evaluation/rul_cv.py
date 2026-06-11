"""
src/ml/evaluation/rul_cv.py
====================================
Phase 6.6: Cross-machine CV for RUL training evaluation.

Background
----------
The Phase 2A evaluation framework's RUL panel uses synthetic
single-machine data; the production RUL predictor trains per-machine
from FAILED events. Neither exercises a held-out machine — there
is no cross-machine generalisation claim. The ``PHASE_6B_DS_AUDIT.md``
flagged this as Dimension 1's "RUL training data has a '100% same
machine' leak in ``_generate_rul_training_data``".

This module adds an evaluation-time cross-machine CV that:

* **Splits by machine** (not by row). The per-row PurgedTimeSeriesCV
  is preserved; the new layer operates one level up, on the
  ``machine_id`` group.
* **Supports three strategies**: ``per_machine`` (the historical
  default, where every machine gets its own model fit on its own
  data — no cross-machine claim), ``leave_one_machine_out``
  (rotation: train on N-1 machines, score on the held-out one),
  and ``k_fold`` (random k-fold over machines).
* **Returns a ``CVStrategyReport``** with per-fold MAE, mean ± std
  MAE across folds, and the best/worst fold pair. The report is
  intended for the model card and the dashboard's "model health"
  section.
* **Does not modify** the production ``RULPredictor`` training
  path. The CV is a one-shot evaluation tool, not a hot-path
  component. The constructor still defaults to ``per_machine`` so
  no production behaviour changes unless the new ``cv_strategy``
  is explicitly requested.

References
----------
* López de Prado (2018). "Advances in Financial Machine Learning."
  Wiley, ch. 7 — the per-row purged k-fold that the existing
  splitter implements. This module layers a *group-level*
  leave-one-out on top of it.
* Arlot & Celisse (2010). "A survey of cross-validation procedures
  for model selection." Statistics Surveys 4:40-79.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


VALID_STRATEGIES = ("per_machine", "leave_one_machine_out", "k_fold")


@dataclass
class CVFoldResult:
    """Outcome of one CV fold."""

    fold: int
    train_machines: tuple[str, ...]
    test_machines: tuple[str, ...]
    train_rows: int
    test_rows: int
    train_mae: float
    test_mae: float
    n_trials_skipped: int = 0  # folds with too few rows to fit
    error: str | None = None


@dataclass
class CVStrategyReport:
    """Aggregated CV result for one model on one (X, y) set."""

    strategy: str
    model_name: str
    metric: str
    fold_results: list[CVFoldResult]
    mean_test_metric: float
    std_test_metric: float
    best_fold: int
    worst_fold: int
    n_skipped_folds: int
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "model_name": self.model_name,
            "metric": self.metric,
            "n_folds": len(self.fold_results),
            "mean_test_metric": float(self.mean_test_metric),
            "std_test_metric": float(self.std_test_metric),
            "best_fold": int(self.best_fold),
            "worst_fold": int(self.worst_fold),
            "n_skipped_folds": int(self.n_skipped_folds),
            "fold_results": [
                {
                    "fold": f.fold,
                    "train_machines": list(f.train_machines),
                    "test_machines": list(f.test_machines),
                    "train_rows": f.train_rows,
                    "test_rows": f.test_rows,
                    "train_mae": float(f.train_mae),
                    "test_mae": float(f.test_mae),
                    "n_trials_skipped": f.n_trials_skipped,
                    "error": f.error,
                }
                for f in self.fold_results
            ],
        }


def _resolve_model(model_name: str):
    """Construct a fresh estimator for the named model."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from xgboost import XGBRegressor

    out: dict[str, object] = {
        "XGBoost": XGBRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "RandomForest": RandomForestRegressor,
    }
    try:
        import lightgbm as lgb

        out["LightGBM"] = lgb.LGBMRegressor
    except ImportError:
        pass

    if model_name not in out:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list(out)}")
    return out[model_name]


def _build_folds(
    machine_ids: np.ndarray,
    strategy: str,
    k: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]]:
    """
    Build the train/test machine-index pairs for the requested
    strategy.

    Returns
    -------
    list[(train_idx, test_idx, train_machines, test_machines)]
        ``train_idx`` and ``test_idx`` are boolean masks over the
        full row array. ``train_machines`` and ``test_machines`` are
        the human-readable group ids (sorted, deduped).
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")
    machines = sorted({str(m) for m in machine_ids})
    if strategy == "per_machine":
        # Every machine is its own fold. Train on the first 80% of
        # that machine's rows (chronological), test on the last 20%.
        # The historical "fit on its own data" path was a vacuous
        # test (per-machine MAE ~ 0 from overfitting). The 80/20
        # split makes per-machine comparable to LOOM.
        masks: list[tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]] = []
        for m in machines:
            machine_idx = np.where(machine_ids == m)[0]
            n = machine_idx.size
            if n < 2:
                continue
            cut = max(1, int(n * 0.8))
            train_idx = machine_idx[:cut]
            test_idx = machine_idx[cut:]
            if test_idx.size == 0:
                test_idx = machine_idx[-1:]
                train_idx = machine_idx[:-1]
            train_mask = np.zeros(machine_ids.shape, dtype=bool)
            test_mask = np.zeros(machine_ids.shape, dtype=bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            masks.append((train_mask, test_mask, (m,), (m,)))
        return masks
    if strategy == "leave_one_machine_out":
        folds: list[tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]] = []
        for held_out in machines:
            train_machines = tuple(m for m in machines if m != held_out)
            test_machines = (held_out,)
            train_mask = np.isin(machine_ids, train_machines)
            test_mask = machine_ids == held_out
            folds.append((train_mask, test_mask, train_machines, test_machines))
        return folds
    if strategy == "k_fold":
        # Deterministic shuffle of the machine list, then k contiguous
        # bins. Each bin is the test set; the rest is the train set.
        rng = np.random.default_rng(seed)
        shuffled = list(machines)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n < k:
            # Fewer machines than folds — fall back to leave-one-out.
            return _build_folds(machine_ids, "leave_one_machine_out", k=k, seed=seed)
        bins: list[list[str]] = [[] for _ in range(k)]
        for i, m in enumerate(shuffled):
            bins[i % k].append(m)
        folds = []
        for i, test_bin in enumerate(bins):
            train_bin = [m for j, b in enumerate(bins) if j != i for m in b]
            train_mask = np.isin(machine_ids, train_bin)
            test_mask = np.isin(machine_ids, test_bin)
            folds.append((train_mask, test_mask, tuple(train_bin), tuple(test_bin)))
        return folds
    raise RuntimeError(f"unreachable: strategy={strategy}")


def cross_validate_rul(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    machine_ids: np.ndarray,
    *,
    strategy: str = "per_machine",
    k: int = 3,
    seed: int = 42,
    min_train_rows: int = 50,
    min_test_rows: int = 10,
) -> CVStrategyReport:
    """
    Run cross-machine CV for one RUL model.

    Parameters
    ----------
    model_name
        Key in the local model registry (``XGBoost``,
        ``GradientBoosting``, ``RandomForest``, ``LightGBM``).
    X, y
        Feature matrix and target vector. ``y`` is the RUL label
        in hours.
    machine_ids
        Per-row machine identifier. Used to build the per-fold
        train/test partition. Required — the function refuses to
        run without it (silent fallback to per-row would
        re-introduce the leak the audit flagged).
    strategy
        ``"per_machine"``, ``"leave_one_machine_out"``, or
        ``"k_fold"``. See module docstring for semantics.
    k
        Number of folds for ``"k_fold"``. Ignored otherwise.
    seed
        RNG seed for ``"k_fold"``'s deterministic shuffle.
    min_train_rows
        Folds with fewer training rows are skipped (recorded as
        ``n_trials_skipped`` in the fold result).
    min_test_rows
        Folds with fewer test rows are skipped for the same
        reason.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mid = np.asarray(machine_ids)

    model_cls = _resolve_model(model_name)
    folds = _build_folds(mid, strategy, k=k, seed=seed)

    fold_results: list[CVFoldResult] = []
    n_skipped = 0
    for i, (train_mask, test_mask, train_machines, test_machines) in enumerate(folds):
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < min_train_rows or n_test < min_test_rows:
            n_skipped += 1
            fold_results.append(
                CVFoldResult(
                    fold=i,
                    train_machines=train_machines,
                    test_machines=test_machines,
                    train_rows=n_train,
                    test_rows=n_test,
                    train_mae=float("nan"),
                    test_mae=float("nan"),
                    n_trials_skipped=1,
                    error="insufficient_rows",
                )
            )
            continue
        try:
            model = model_cls()
            model.fit(X_arr[train_mask], y_arr[train_mask])
            train_pred = model.predict(X_arr[train_mask])
            test_pred = model.predict(X_arr[test_mask])
            train_mae = float(np.mean(np.abs(y_arr[train_mask] - train_pred)))
            test_mae = float(np.mean(np.abs(y_arr[test_mask] - test_pred)))
        except Exception as exc:  # noqa: BLE001
            fold_results.append(
                CVFoldResult(
                    fold=i,
                    train_machines=train_machines,
                    test_machines=test_machines,
                    train_rows=n_train,
                    test_rows=n_test,
                    train_mae=float("nan"),
                    test_mae=float("nan"),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        fold_results.append(
            CVFoldResult(
                fold=i,
                train_machines=train_machines,
                test_machines=test_machines,
                train_rows=n_train,
                test_rows=n_test,
                train_mae=train_mae,
                test_mae=test_mae,
            )
        )

    valid = [f for f in fold_results if f.test_mae == f.test_mae]  # NaN check
    if not valid:
        return CVStrategyReport(
            strategy=strategy,
            model_name=model_name,
            metric="mae",
            fold_results=fold_results,
            mean_test_metric=float("nan"),
            std_test_metric=float("nan"),
            best_fold=-1,
            worst_fold=-1,
            n_skipped_folds=n_skipped,
            extra={"reason": "all_folds_skipped_or_failed"},
        )
    test_maes = np.asarray([f.test_mae for f in valid], dtype=np.float64)
    best_idx = int(np.argmin(test_maes))
    worst_idx = int(np.argmax(test_maes))
    return CVStrategyReport(
        strategy=strategy,
        model_name=model_name,
        metric="mae",
        fold_results=fold_results,
        mean_test_metric=float(np.mean(test_maes)),
        std_test_metric=float(np.std(test_maes)),
        best_fold=valid[best_idx].fold,
        worst_fold=valid[worst_idx].fold,
        n_skipped_folds=n_skipped,
        extra={"n_valid_folds": len(valid)},
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m src.ml.evaluation.rul_cv``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Cross-machine RUL CV")
    parser.add_argument("--model", default="XGBoost")
    parser.add_argument(
        "--strategy",
        default="leave_one_machine_out",
        choices=list(VALID_STRATEGIES),
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-machines", type=int, default=4)
    parser.add_argument("--n-per-machine", type=int, default=200)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    n_total = args.n_machines * args.n_per_machine
    X = rng.normal(0, 1, size=(n_total, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 0.3]) + rng.normal(0, 0.1, n_total)
    machine_ids = np.array(
        [f"M-{i // args.n_per_machine:02d}" for i in range(n_total)]
    )
    report = cross_validate_rul(
        args.model,
        X,
        y,
        machine_ids,
        strategy=args.strategy,
        k=args.k,
        seed=args.seed,
    )
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
