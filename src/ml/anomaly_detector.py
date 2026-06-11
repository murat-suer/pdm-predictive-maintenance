import hashlib
import logging
import os
import time

import joblib
import numpy as np
import shap
from sklearn.ensemble import IsolationForest

from src.ml.feature_engineering import compute_features, filter_calibration_data
from src.ml.model_card import training_window_from_timestamps, write_model_card
from src.ml.model_store import MODEL_ARTIFACT_VERSION, get_model_store

logger = logging.getLogger(__name__)

CONTAMINATION = float(os.environ.get("ML_CONTAMINATION", "0.05"))


class AnomalyDetector:
    def __init__(self, machine_id: str, contamination: float | None = None):
        self.machine_id = machine_id
        self.model: IsolationForest | None = None
        self.explainer = None
        self.feature_names: list = []
        self._contamination = contamination if contamination is not None else CONTAMINATION
        store = get_model_store()
        self._model_path = str(store / f"{machine_id}_anomaly.joblib")
        if os.path.exists(self._model_path):
            self._load_model()

    def train(self, raw_df) -> None:
        train_df = filter_calibration_data(
            raw_df,
            exclude_phases=["ANOMALY", "FAILED", "IDLE", "CALIBRATING", "CANARY_EVENT"],
        )
        if len(train_df) < 50:
            logger.warning(f"{self.machine_id}: insufficient training data ({len(train_df)} rows)")
            return
        from src.data_generator.machines import MACHINE_CONFIGS
        features_df = compute_features(train_df, MACHINE_CONFIGS.get(self.machine_id))
        feature_cols = [c for c in features_df.columns if c != "timestamp"]
        self.feature_names = feature_cols
        X = features_df.select(feature_cols).fill_null(0.0).to_numpy()
        if X.shape[0] < 50:
            raise RuntimeError(
                f"{self.machine_id}: post-feature training matrix is too small "
                f"({X.shape[0]} rows x {X.shape[1]} features). Minimum is 50 rows."
            )
        self.model = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
        )
        self.model.fit(X)
        self.explainer = shap.TreeExplainer(self.model)
        timestamps = train_df["timestamp"].to_list() if "timestamp" in train_df.columns else []
        machines_covered = sorted({
            str(mid) for mid in (
                raw_df["machine_id"].to_list() if "machine_id" in raw_df.columns else [self.machine_id]
            ) if mid
        })
        if not machines_covered:
            machines_covered = [self.machine_id]
        self._save_model(
            len(X),
            training_window_from_timestamps(timestamps),
            machines_covered=machines_covered,
        )
        logger.info(f"{self.machine_id}: AnomalyDetector trained on {len(X)} samples")

    def predict(self, features_row: dict) -> dict:
        if self.model is None:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "top_contributing_sensor": None,
                "shap_values": {},
            }
        X = np.array([[features_row.get(f, 0.0) for f in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0)
        t0 = time.perf_counter()
        raw_pred = self.model.predict(X)
        score_raw = self.model.score_samples(X)[0]
        t1 = time.perf_counter()
        anomaly_score = float(max(0.0, min(1.0, -float(score_raw))))
        is_anomaly = bool(raw_pred[0] == -1)
        logger.debug(f"{self.machine_id}: IF predict in {(t1 - t0) * 1000:.1f}ms")
        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "top_contributing_sensor": None,
            "shap_values": {},
        }
        if is_anomaly and self.explainer is not None:
            result.update(self._compute_shap(X))
        return result

    def _compute_shap(self, X: np.ndarray) -> dict:
        t0 = time.perf_counter()
        try:
            shap_values = self.explainer.shap_values(X)
            shap_row = shap_values[0] if shap_values.ndim > 1 else shap_values
            shap_dict = {feat: round(float(val), 6) for feat, val in zip(self.feature_names, shap_row, strict=False)}
            value_feats = {feat: abs(val) for feat, val in shap_dict.items() if feat.endswith("_value")}
            if value_feats:
                top_feat = max(value_feats, key=value_feats.get)
                top_sensor = top_feat.replace("_value", "")
            else:
                top_sensor = None
            t1 = time.perf_counter()
            logger.debug(f"{self.machine_id}: SHAP in {(t1 - t0) * 1000:.1f}ms")
            return {
                "top_contributing_sensor": top_sensor,
                "shap_values": shap_dict,
            }
        except Exception as e:
            logger.error(f"{self.machine_id}: SHAP failed: {e}")
            return {"top_contributing_sensor": None, "shap_values": {}}

    def _save_model(self, training_rows: int, training_window: dict, machines_covered: list[str] | None = None):
        store = get_model_store()
        os.makedirs(store, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "explainer": self.explainer,
                "feature_names": self.feature_names,
                "artifact_version": MODEL_ARTIFACT_VERSION,
            },
            self._model_path,
        )
        hasher = hashlib.sha256()
        with open(self._model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        with open(f"{self._model_path}.sha256", "w") as f:
            f.write(hasher.hexdigest())
        write_model_card(
            self._model_path,
            model_kind="anomaly_detector",
            machine_id=self.machine_id,
            feature_list=self.feature_names,
            hyperparameters={
                "contamination": self._contamination,
                "n_estimators": 100,
                "artifact_version": MODEL_ARTIFACT_VERSION,
            },
            training_rows=training_rows,
            training_window=training_window,
            machines_covered=machines_covered,
        )
        logger.info(f"Saved: {self._model_path}")

    def _load_model(self):
        try:
            hash_path = f"{self._model_path}.sha256"
            if not os.path.exists(hash_path) and os.path.exists(self._model_path):
                logger.warning(f"Hash file missing for {self._model_path}; generating bootstrap hash.")
                hasher = hashlib.sha256()
                with open(self._model_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                with open(hash_path, "w") as f:
                    f.write(hasher.hexdigest())
            if not os.path.exists(hash_path):
                raise RuntimeError(f"Model file integrity check failed for {self._model_path}")
            with open(hash_path) as f:
                stored_hash = f.read().strip()
            hasher = hashlib.sha256()
            with open(self._model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != stored_hash:
                raise RuntimeError(f"Model file integrity check failed for {self._model_path}")
            data = joblib.load(self._model_path)
            if data.get("artifact_version") != MODEL_ARTIFACT_VERSION:
                logger.warning(f"{self.machine_id}: stale anomaly model artifact; fresh calibration required")
                return
            self.model = data["model"]
            self.explainer = data.get("explainer")
            self.feature_names = data.get("feature_names", [])
            card_path = self._model_path.replace(".joblib", ".model_card.json")
            if not os.path.exists(card_path):
                write_model_card(
                    self._model_path,
                    model_kind="anomaly_detector",
                    machine_id=self.machine_id,
                    feature_list=self.feature_names,
                    hyperparameters={
                        "contamination": self._contamination,
                        "n_estimators": 100,
                        "artifact_version": MODEL_ARTIFACT_VERSION,
                    },
                    training_rows=None,
                )
            logger.info(f"Loaded: {self._model_path}")
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to load {self._model_path}: {e}")
