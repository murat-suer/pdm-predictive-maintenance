from __future__ import annotations

import logging

import polars as pl

from src.data_generator.machines import MACHINE_CONFIGS
from src.ml.anomaly_detector import AnomalyDetector
from src.ml.buffer_manager import generate_synthetic_calibration_buffer
from src.ml.calibration_buffer import CalibrationBuffer
from src.ml.rul_predictor import RULPredictor

logger = logging.getLogger(__name__)

CALIBRATION_MIN = 10800

__all__ = ["CalibrationBuffer", "PipelineState", "train_all_models", "build_detectors", "build_predictors", "all_models_trained"]


class PipelineState:
    def __init__(self):
        self.mode: str = "CALIBRATING"

    def is_production(self) -> bool:
        return self.mode == "PRODUCTION"

    def set_production(self) -> None:
        self.mode = "PRODUCTION"

    def set_calibrating(self) -> None:
        self.mode = "CALIBRATING"


def train_all_models(detectors: dict, predictors: dict, buffers: dict):
    for machine_id, buffer in buffers.items():
        if len(buffer) < 50:
            buffer = generate_synthetic_calibration_buffer(machine_id)
            buffers[machine_id] = buffer
            logger.warning(
                f"{machine_id}: insufficient healthy stream history; "
                f"using synthetic nominal calibration buffer ({len(buffer)} readings)"
            )
        try:
            df = pl.DataFrame({
                "timestamp": [b.get("timestamp", "") for b in buffer],
                "sensor_name": [b.get("sensor_name", "") for b in buffer],
                "value": [float(b.get("value", 0)) for b in buffer],
                "machine_phase": [b.get("phase", "HEALTHY") for b in buffer],
                "upstream_effect": [b.get("upstream_effect", False) for b in buffer],
            })
            detectors[machine_id].train(df)
            logger.info(f"Trained AnomalyDetector: {machine_id}")
            try:
                from src.database.connection import get_db_context
                with get_db_context() as db:
                    X, y = predictors[machine_id]._generate_rul_training_data(machine_id, db)
                    if X is not None and len(X) >= 50:
                        predictors[machine_id].train(X, y, predictors[machine_id].feature_names)
                        logger.info(f"[ML] RUL XGBoost trained: {machine_id} ({len(X)} samples)")
                    else:
                        logger.info(f"[ML] RUL fallback mode: {machine_id} (insufficient data)")
            except Exception as e:
                logger.error(f"[ML] RUL training failed {machine_id}: {e}")
        except Exception as e:
            logger.error(f"Training failed for {machine_id}: {e}")


def build_detectors() -> dict:
    detectors = {}
    for machine_id in MACHINE_CONFIGS:
        detectors[machine_id] = AnomalyDetector(machine_id)
    return detectors


def build_predictors() -> dict:
    predictors = {}
    for machine_id in MACHINE_CONFIGS:
        weibull = MACHINE_CONFIGS[machine_id]["weibull"]
        predictors[machine_id] = RULPredictor(
            machine_id,
            beta=weibull["beta"],
            eta=weibull["eta"],
        )
    return predictors


def all_models_trained(detectors: dict) -> bool:
    return all(detectors[mid].model is not None for mid in MACHINE_CONFIGS)
