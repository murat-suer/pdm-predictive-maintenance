from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.ml.model_store import get_model_store

logger = logging.getLogger(__name__)

KNOWN_LIMITATIONS = [
    "Simulation-trained - not validated on production-grade sensor data",
    "Requires baseline calibration; out-of-distribution machines may underperform",
]

# Minimum total rows required to attempt a temporal holdout split.
# With 80/20 train/test, 100 rows gives at least 20 test rows — enough
# for a meaningful MAE estimate without risking near-zero test sets.
_MIN_SAMPLES_FOR_HOLDOUT = 100


def write_model_card(
    artifact_path: str | Path,
    *,
    model_kind: str,
    machine_id: str,
    feature_list: list[str],
    hyperparameters: dict[str, Any],
    training_rows: int | None,
    training_window: dict[str, str | None] | None = None,
    metrics: dict[str, Any] | None = None,
    machines_covered: list[str] | None = None,
) -> Path:
    artifact = Path(artifact_path)
    card_path = artifact.with_suffix(".model_card.json")
    card_path.write_text(
        json.dumps(
            build_model_card(
                artifact,
                model_kind=model_kind,
                machine_id=machine_id,
                feature_list=feature_list,
                hyperparameters=hyperparameters,
                training_rows=training_rows,
                training_window=training_window,
                metrics=metrics,
                machines_covered=machines_covered,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return card_path


def build_model_card(
    artifact_path: str | Path,
    *,
    model_kind: str,
    machine_id: str,
    feature_list: list[str],
    hyperparameters: dict[str, Any],
    training_rows: int | None,
    training_window: dict[str, str | None] | None = None,
    metrics: dict[str, Any] | None = None,
    machines_covered: list[str] | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    window = training_window or {"from": None, "to": None}
    card = {
        "name": f"{model_kind}_{machine_id}",
        "version": "2.1.0",
        "trained_at": datetime.fromtimestamp(artifact.stat().st_mtime, tz=UTC).isoformat(),
        "artifact": artifact.name,
        "training_window": window,
        "training_rows": training_rows,
        "machines_covered": list(machines_covered) if machines_covered else [machine_id],
        "training_data_signature": _training_signature(model_kind, machine_id, feature_list, training_rows, window),
        "artifact_sha256": _file_sha256(artifact),
        "feature_list": feature_list,
        "hyperparameters": hyperparameters,
        "metrics": metrics or _unmeasured_metrics(model_kind),
        "intended_use": _intended_use(model_kind, machine_id),
        "known_limitations": KNOWN_LIMITATIONS,
    }
    return card


def read_model_cards(model_store: str | Path | None = None) -> list[dict[str, Any]]:
    store = Path(model_store) if model_store else get_model_store()
    cards = []
    for path in sorted(store.glob("*.model_card.json")):
        try:
            cards.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return cards


def training_window_from_timestamps(timestamps: list[str]) -> dict[str, str | None]:
    clean = sorted(str(timestamp) for timestamp in timestamps if timestamp)
    return {"from": clean[0] if clean else None, "to": clean[-1] if clean else None}


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(8192), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _training_signature(
    model_kind: str,
    machine_id: str,
    feature_list: list[str],
    training_rows: int | None,
    training_window: dict[str, str | None],
) -> str:
    payload = json.dumps(
        {
            "model_kind": model_kind,
            "machine_id": machine_id,
            "feature_list": feature_list,
            "training_rows": training_rows,
            "training_window": training_window,
        },
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def compute_rul_holdout_metrics(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Compute held-out MAE and RMSE for a RUL predictor via a temporal split.

    Splits X/y into the first 80 % (train) and last 20 % (test) by row order.
    Row order is assumed to be chronological, as guaranteed by the callers in
    :class:`src.ml.rul_predictor.RULPredictor`.  This is an honest single
    out-of-sample evaluation — not a random split — preserving the
    project's "no leak" property.

    Parameters
    ----------
    X:
        Feature matrix, already in chronological order.
    y:
        RUL target vector, same order.

    Returns
    -------
    dict with keys:
        ``holdout_mae_hours``  — float, mean absolute error on the test split.
        ``holdout_rmse_hours`` — float, root-mean-squared error on the test split.
        ``holdout_n_test``     — int, number of test rows used.
        ``holdout_n_train``    — int, number of train rows used.
        ``split_method``       — str, always ``"temporal_80_20"``.
        ``status``             — str, ``"measured"`` on success or ``"insufficient_samples"``
                                 if there were not enough rows.
        ``reason``             — str | None, non-None only when status != "measured".
    """
    n = len(X)
    if n < _MIN_SAMPLES_FOR_HOLDOUT:
        return {
            "status": "insufficient_samples",
            "reason": f"need >= {_MIN_SAMPLES_FOR_HOLDOUT} rows, got {n}",
            "holdout_mae_hours": None,
            "holdout_rmse_hours": None,
            "holdout_n_test": None,
            "holdout_n_train": None,
            "split_method": "temporal_80_20",
        }

    from xgboost import XGBRegressor

    cut = max(1, int(round(n * 0.80)))
    # Guard: ensure at least 10 rows on each side
    cut = max(10, min(cut, n - 10))

    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    try:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        holdout_mae = float(np.mean(np.abs(residuals)))
        holdout_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        return {
            "status": "measured",
            "reason": None,
            "holdout_mae_hours": round(holdout_mae, 4),
            "holdout_rmse_hours": round(holdout_rmse, 4),
            "holdout_n_test": int(len(y_test)),
            "holdout_n_train": int(len(y_train)),
            "split_method": "temporal_80_20",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("RUL holdout metric computation failed: %s", exc)
        return {
            "status": "computation_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "holdout_mae_hours": None,
            "holdout_rmse_hours": None,
            "holdout_n_test": None,
            "holdout_n_train": None,
            "split_method": "temporal_80_20",
        }


def _unmeasured_metrics(model_kind: str) -> dict[str, Any]:
    if model_kind == "rul_predictor":
        return {
            "status": "insufficient_samples",
            "reason": "no training data available at card-creation time",
            "holdout_mae_hours": None,
            "holdout_rmse_hours": None,
            "holdout_n_test": None,
            "holdout_n_train": None,
            "split_method": "temporal_80_20",
        }
    return {
        "status": "not_measured_in_current_simulation",
        "note": f"{model_kind} model cards expose provenance; holdout performance metrics require labeled evaluation data.",
        "roc_auc": None,
        "precision_at_recall_90": None,
        "per_fault_auc": {},
    }


def _intended_use(model_kind: str, machine_id: str) -> str:
    if model_kind == "rul_predictor":
        return f"Remaining useful life estimation for simulated sensor streams on {machine_id}."
    return f"Real-time anomaly detection on simulated sensor streams for {machine_id}."
