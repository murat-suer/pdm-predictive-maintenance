from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.data_generator.machines import MACHINE_CONFIGS
from src.ml.buffer_manager import extract_latest_features
from src.ml.calibration import PipelineState

logger = logging.getLogger(__name__)

# ─── PINN Integration Constants ─────────────────────────────────────────────
# Threshold for model selection: use PINN when dataset has enough samples
# for neural network training to be worthwhile.
PINN_MIN_SAMPLES = 500
# Default physics loss weight
DEFAULT_LAMBDA_PHYSICS = 0.1
# Default training epochs for PINN
DEFAULT_PINN_EPOCHS = 100
# Default learning rate for PINN training
DEFAULT_PINN_LR = 0.001
# Hidden dimension for PINN network
DEFAULT_PINN_HIDDEN_DIM = 64


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    model_type: str
    n_samples: int
    n_features: int
    n_epochs: int = 0
    final_loss: float = 0.0
    final_data_loss: float = 0.0
    final_physics_loss: float = 0.0
    loss_history: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "model_type": self.model_type,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_epochs": self.n_epochs,
            "final_loss": self.final_loss,
            "final_data_loss": self.final_data_loss,
            "final_physics_loss": self.final_physics_loss,
            "loss_history_length": len(self.loss_history),
        }


def select_model_type(n_samples: int, threshold: int = PINN_MIN_SAMPLES) -> str:
    """Select model type based on dataset size.

    PINNs are preferred for larger datasets where neural networks can
    leverage the physics-informed constraints effectively. XGBoost is
    preferred for smaller datasets where tree-based methods generalize
    better with limited data.

    Parameters
    ----------
    n_samples : int
        Number of training samples available.
    threshold : int
        Minimum samples to prefer PINN over XGBoost.

    Returns
    -------
    str
        "pinn" or "xgboost"
    """
    if n_samples >= threshold:
        return "pinn"
    return "xgboost"


def train_pinn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timestamps: np.ndarray,
    eta: float,
    beta: float,
    n_epochs: int = DEFAULT_PINN_EPOCHS,
    lambda_physics: float = DEFAULT_LAMBDA_PHYSICS,
    hidden_dim: int = DEFAULT_PINN_HIDDEN_DIM,
    learning_rate: float = DEFAULT_PINN_LR,
) -> tuple[Any, TrainingMetrics]:
    """Train PINN model with physics-informed loss.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_train : np.ndarray
        Target RUL values of shape (n_samples,).
    timestamps : np.ndarray
        Timestamps (operational time) for each sample, shape (n_samples,).
    eta : float
        Weibull scale parameter (characteristic life).
    beta : float
        Weibull shape parameter.
    n_epochs : int
        Number of training epochs.
    lambda_physics : float
        Weight for the physics-informed loss term.
    hidden_dim : int
        Hidden layer dimension for the neural network.
    learning_rate : float
        Learning rate for Adam optimizer.

    Returns
    -------
    tuple[PINNRULPredictor, TrainingMetrics]
        Trained model and training metrics. The model is called as
        ``model(X, t)`` — machine age is part of the input contract.
    """
    import torch

    from src.ml.pinn_rul import PhysicsInformedLoss, PINNRULPredictor

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32)

    n_samples, n_features = X_train.shape

    # Create model and loss
    model = PINNRULPredictor(input_dim=n_features, hidden_dim=hidden_dim)
    loss_fn = PhysicsInformedLoss(lambda_physics=lambda_physics)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Convert to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    t_tensor = torch.tensor(timestamps, dtype=torch.float32, requires_grad=True)

    loss_history: list[float] = []
    final_data_loss = 0.0
    final_physics_loss = 0.0

    # Training loop
    model.train()
    for _epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass — t enters the graph so the physics loss can take
        # d(RUL)/dt via autograd instead of the finite-difference fallback
        rul_pred = model(X_tensor, t_tensor)

        # Compute losses
        data_loss = torch.nn.functional.mse_loss(rul_pred, y_tensor)

        # Physics loss components
        phys_loss = loss_fn.physics_loss(t_tensor, rul_pred, eta, beta)

        # Combined loss
        total_loss = data_loss + lambda_physics * phys_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)
        final_data_loss = data_loss.item()
        final_physics_loss = phys_loss.item()

    model.eval()

    metrics = TrainingMetrics(
        model_type="pinn",
        n_samples=n_samples,
        n_features=n_features,
        n_epochs=n_epochs,
        final_loss=loss_history[-1] if loss_history else 0.0,
        final_data_loss=final_data_loss,
        final_physics_loss=final_physics_loss,
        loss_history=loss_history,
        extra={
            "eta": eta,
            "beta": beta,
            "lambda_physics": lambda_physics,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
        },
    )

    logger.info(
        f"PINN training complete: {n_epochs} epochs, "
        f"final_loss={metrics.final_loss:.4f}, "
        f"data_loss={final_data_loss:.4f}, "
        f"physics_loss={final_physics_loss:.4f}"
    )

    return model, metrics


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs: Any,
) -> tuple[Any, TrainingMetrics]:
    """Train XGBoost model for RUL prediction.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix.
    y_train : np.ndarray
        Target RUL values.
    **kwargs
        Additional keyword arguments passed to XGBRegressor.

    Returns
    -------
    tuple[XGBRegressor, TrainingMetrics]
        Trained model and training metrics.
    """
    from xgboost import XGBRegressor

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    n_samples, n_features = X_train.shape

    # Default XGBoost params
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Compute training metrics
    y_pred = model.predict(X_train)
    mse = float(np.mean((y_train - y_pred) ** 2))

    metrics = TrainingMetrics(
        model_type="xgboost",
        n_samples=n_samples,
        n_features=n_features,
        n_epochs=params.get("n_estimators", 200),
        final_loss=mse,
        final_data_loss=mse,
        final_physics_loss=0.0,
        extra={"params": {k: v for k, v in params.items() if k != "n_jobs"}},
    )

    logger.info(f"XGBoost training complete: MSE={mse:.4f}")
    return model, metrics


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timestamps: np.ndarray | None = None,
    eta: float = 500.0,
    beta: float = 2.0,
    model_type: str | None = None,
    n_epochs: int = DEFAULT_PINN_EPOCHS,
    lambda_physics: float = DEFAULT_LAMBDA_PHYSICS,
    **kwargs: Any,
) -> tuple[Any, TrainingMetrics]:
    """Unified training interface that selects and trains the appropriate model.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix.
    y_train : np.ndarray
        Target RUL values.
    timestamps : np.ndarray | None
        Timestamps for PINN training. Required if model_type is "pinn" or auto.
    eta : float
        Weibull scale parameter for PINN physics loss.
    beta : float
        Weibull shape parameter for PINN physics loss.
    model_type : str | None
        "pinn", "xgboost", or None for auto-selection based on data size.
    n_epochs : int
        Number of epochs for PINN training.
    lambda_physics : float
        Physics loss weight for PINN training.
    **kwargs
        Additional arguments passed to the model trainer.

    Returns
    -------
    tuple[model, TrainingMetrics]
        Trained model and metrics.
    """
    n_samples = X_train.shape[0]

    if model_type is None:
        model_type = select_model_type(n_samples)
        logger.info(f"Auto-selected model type: {model_type} (n_samples={n_samples})")

    if model_type == "pinn":
        if timestamps is None:
            raise ValueError("timestamps required for PINN training")
        return train_pinn_model(
            X_train=X_train,
            y_train=y_train,
            timestamps=timestamps,
            eta=eta,
            beta=beta,
            n_epochs=n_epochs,
            lambda_physics=lambda_physics,
            **kwargs,
        )
    elif model_type == "xgboost":
        return train_xgboost_model(X_train=X_train, y_train=y_train, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'pinn' or 'xgboost'.")

CONSUMER_GROUP = "pm-ml-group"
CONSUMER_NAME = "pm-ml-1"
STREAM_KEY = "sensor_data_stream"
PREDICTION_WARMUP_MIN = 96
PREDICTION_GRACE_CYCLES = 6
ANOMALY_CONFIRM_CYCLES = 2
ANOMALY_CONFIRM_WINDOW_S = 90
HEALTHY_ALARM_SCORE_MIN = 0.70
# Backpressure: when the consumer lags the stream, stale events only feed
# the rolling buffers — inference runs on near-real-time data only.
STALE_EVENT_SKIP_S = 90.0


def process_event(
    event_id,
    fields: dict,
    detectors,
    predictors,
    mhi_calc,
    buffers: dict,
    r,
    machine_states: dict,
    pipeline_state: PipelineState,
    cusum_detectors: dict | None = None,
    fault_classifier=None,
):
    payload = json.loads(fields.get(b"payload", b"{}"))
    ts_raw = fields.get(b"timestamp", b"")
    payload["timestamp"] = ts_raw.decode() if isinstance(ts_raw, bytes) else str(ts_raw)
    machine_id = payload.get("machine_id", "")
    phase = payload.get("phase", "HEALTHY")
    upstream = payload.get("upstream_effect", False)
    if phase == "CANARY_EVENT":
        return
    if payload.get("present", True) is False:
        _record_sensor_dropout(payload, r, machine_states)
        return
    if phase == "FAILED" and machine_id in machine_states:
        prev_phase = machine_states[machine_id].get("last_phase")
        if prev_phase != "FAILED":
            machine_states[machine_id]["emergency_stop_count"] += 1
            _update_conformal_from_failure(
                predictors.get(machine_id),
                machine_states[machine_id],
                payload.get("timestamp", ""),
            )
        machine_states[machine_id]["last_phase"] = "FAILED"
    elif machine_id in machine_states:
        machine_states[machine_id]["last_phase"] = phase
    if machine_id in buffers:
        buffers[machine_id].append(payload)
        if len(buffers[machine_id]) > 500:
            buffers[machine_id] = buffers[machine_id][-500:]
    if upstream:
        return
    if fields.get(b"warmup") == b"true" or payload.get("warmup"):
        return
    try:
        event_at = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
        if (datetime.now(UTC) - event_at).total_seconds() > STALE_EVENT_SKIP_S:
            return
    except (KeyError, TypeError, ValueError):
        pass
    if pipeline_state.is_production() and machine_id in detectors:
        detector = detectors[machine_id]
        if detector.model is not None and len(buffers.get(machine_id, [])) >= PREDICTION_WARMUP_MIN:
            features = extract_latest_features(buffers[machine_id], MACHINE_CONFIGS.get(machine_id))
            if features:
                det_result = detector.predict(features)
                # The payload object is the buffer entry (appended above by
                # reference) — flag it so MHI's anomaly-rate term sees it.
                payload["is_anomaly"] = bool(det_result["is_anomaly"])
                machine_state = machine_states.setdefault(machine_id, {"emergency_stop_count": 0})
                timestamp = payload.get("timestamp", "")

                rul_result = None
                predictor = predictors.get(machine_id)
                if predictor is not None:
                    try:
                        rul_result = predictor.predict(
                            features=features,
                            phase=phase,
                            emergency_stop_count=machine_state.get("emergency_stop_count", 0),
                            degradation_level=_machine_degradation(
                                payload, buffers.get(machine_id, [])
                            ),
                        )
                        remember_rul_prediction(machine_state, rul_result, timestamp)
                    except Exception as exc:
                        logger.debug(f"RUL prediction failed for {machine_id}: {exc}")

                if det_result["is_anomaly"]:
                    logger.info(f"Anomaly detected: {machine_id} score={det_result['anomaly_score']}")
                    _persist_and_publish_anomaly(
                        machine_id, payload, det_result, features,
                        buffers, r, machine_state, phase, timestamp,
                        fault_classifier, rul_result,
                    )

                if mhi_calc is not None:
                    _update_mhi_throttled(
                        machine_id, phase, payload, det_result, rul_result,
                        mhi_calc, buffers, machine_state,
                    )


def _machine_degradation(payload: dict, buffer: list) -> float | None:
    """Machine-level degradation: the worst sensor in the recent window.

    Payload degradation is PER-SENSOR (each event is one reading); using it
    directly made RUL flicker between the healthy and the failing sensor's
    estimate. A machine fails by its worst component, so the binding level
    is the max over the last few readings (~all sensors at 5/machine).
    """
    levels = [
        r.get("degradation_level")
        for r in buffer[-25:]
        if r.get("degradation_level") is not None
    ]
    current = payload.get("degradation_level")
    if current is not None:
        levels.append(current)
    return max(levels) if levels else None


def _persist_and_publish_anomaly(
    machine_id, payload, det_result, features,
    buffers, r, machine_state, phase, timestamp,
    fault_classifier, rul_result=None,
):
    """Confirmed anomalies are written to anomaly_log and broadcast on
    anomaly_stream, where the decision service picks them up."""
    from src.ml.event_publisher import publish_anomaly_event
    from src.ml.persistence import update_sensor_reading_anomaly, write_anomaly_log

    if in_prediction_grace(machine_state, timestamp):
        return
    if not is_alarm_candidate(phase, True, det_result["anomaly_score"]):
        return
    if not anomaly_is_confirmed(machine_state, timestamp):
        return
    try:
        snapshot = latest_sensor_snapshot(buffers.get(machine_id, []))
        anomaly_id, is_new, _fault = write_anomaly_log(
            machine_id, payload, det_result,
            features=features, sensor_snapshot=snapshot,
            fault_classifier=fault_classifier,
        )
        update_sensor_reading_anomaly(machine_id, det_result["anomaly_score"])
        if is_new:
            rul_hours = rul_result.get("rul_hours") if rul_result else None
            publish_anomaly_event(
                r, machine_id, anomaly_id, det_result, payload, rul_hours=rul_hours
            )
            logger.info(f"Anomaly persisted and published: {machine_id} anomaly_id={anomaly_id}")
    except Exception as exc:
        logger.error(f"Anomaly persistence failed for {machine_id}: {exc}")


MHI_UPDATE_INTERVAL_S = 30.0


def _update_mhi_throttled(
    machine_id, phase, payload, det_result, rul_result,
    mhi_calc, buffers, machine_state,
):
    """Write a MachineHealthScore row at most every MHI_UPDATE_INTERVAL_S."""
    import time as _time

    from src.ml.persistence import update_mhi

    now = _time.monotonic()
    last = machine_state.get("last_mhi_monotonic", 0.0)
    if now - last < MHI_UPDATE_INTERVAL_S:
        return
    machine_state["last_mhi_monotonic"] = now
    try:
        update_mhi(machine_id, phase, payload, det_result, rul_result, mhi_calc, buffers)
    except Exception as exc:
        logger.error(f"MHI update failed for {machine_id}: {exc}")


def _record_sensor_dropout(payload, r, machine_states: dict) -> None:
    try:
        machine_id = payload.get("machine_id", "")
        sensor_name = payload.get("sensor_name", "")
        timestamp = payload.get("timestamp", "")
        operational_seconds = float(payload.get("operational_seconds", 0.0))
        phase = payload.get("phase", "HEALTHY")
        if machine_id and machine_id in machine_states:
            ms = machine_states[machine_id]
            ms["sensor_dropout_count"] = int(ms.get("sensor_dropout_count", 0)) + 1
            ms["last_dropout_at"] = timestamp
            ms["last_dropout_sensor"] = sensor_name
        try:
            r.xadd(
                "sensor_dropout_stream",
                {
                    "machine_id": machine_id,
                    "sensor_name": sensor_name,
                    "operational_seconds": f"{operational_seconds:.1f}",
                    "phase": phase,
                    "timestamp": timestamp,
                },
                maxlen=5000,
            )
        except Exception as exc:
            logger.debug(f"sensor_dropout_stream xadd failed: {exc}")
    except Exception as exc:
        logger.debug(f"_record_sensor_dropout failed: {exc}")


def _update_conformal_from_failure(predictor, machine_state: dict, failed_at: str) -> None:
    prediction = machine_state.pop("latest_rul_prediction", None)
    if predictor is None or not prediction or not failed_at:
        return
    try:
        predicted_at = datetime.fromisoformat(prediction["timestamp"].replace("Z", "+00:00"))
        failure_at = datetime.fromisoformat(failed_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return
    actual_hours = max(0.0, (failure_at - predicted_at).total_seconds() / 3600.0)
    predictor.update_conformal(prediction["predicted_hours"], actual_hours)


def in_prediction_grace(machine_state: dict, timestamp: str) -> bool:
    if not timestamp:
        return True
    cycle_timestamps = machine_state.setdefault("prediction_cycle_timestamps", [])
    if not cycle_timestamps or cycle_timestamps[-1] != timestamp:
        cycle_timestamps.append(timestamp)
        del cycle_timestamps[:-PREDICTION_GRACE_CYCLES]
        machine_state["prediction_cycle_count"] = machine_state.get("prediction_cycle_count", 0) + 1
    return machine_state["prediction_cycle_count"] <= PREDICTION_GRACE_CYCLES


def is_alarm_candidate(phase: str, is_anomaly: bool, anomaly_score: float) -> bool:
    if not is_anomaly:
        return False
    if phase == "HEALTHY":
        return anomaly_score >= HEALTHY_ALARM_SCORE_MIN
    return True


def anomaly_is_confirmed(machine_state: dict, timestamp: str) -> bool:
    if not timestamp:
        return False
    history = machine_state.setdefault("anomaly_cycle_timestamps", [])
    if history and history[-1] == timestamp:
        return len(history) >= ANOMALY_CONFIRM_CYCLES
    try:
        detected_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    previous = machine_state.get("last_anomaly_cycle_at")
    if previous is not None:
        elapsed = (detected_at - previous).total_seconds()
        if elapsed < 0 or elapsed > ANOMALY_CONFIRM_WINDOW_S:
            history.clear()
    history.append(timestamp)
    del history[:-ANOMALY_CONFIRM_CYCLES]
    machine_state["last_anomaly_cycle_at"] = detected_at
    return len(history) >= ANOMALY_CONFIRM_CYCLES


def remember_rul_prediction(machine_state: dict, rul_result: dict | None, timestamp: str) -> None:
    if not rul_result or rul_result.get("rul_hours") is None or not timestamp:
        return
    machine_state["latest_rul_prediction"] = {
        "predicted_hours": float(rul_result["rul_hours"]),
        "timestamp": timestamp,
    }


def latest_sensor_snapshot(buffer: list) -> dict:
    sensor_readings = {}
    for reading in reversed(buffer):
        sensor_name = reading.get("sensor_name")
        sensor_value = reading.get("value")
        if not sensor_name or sensor_name in sensor_readings or sensor_value is None:
            continue
        try:
            sensor_readings[sensor_name] = float(sensor_value)
        except (TypeError, ValueError):
            continue
    return sensor_readings
