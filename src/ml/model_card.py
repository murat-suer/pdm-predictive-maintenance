from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.ml.model_store import get_model_store

KNOWN_LIMITATIONS = [
    "Simulation-trained - not validated on production-grade sensor data",
    "Requires baseline calibration; out-of-distribution machines may underperform",
]


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


def _unmeasured_metrics(model_kind: str) -> dict[str, Any]:
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
