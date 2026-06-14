import hashlib
import logging
import os

import joblib
import numpy as np
import scipy.stats as stats

from src.ml.conformal_rul import ConformalRUL
from src.ml.model_card import write_model_card
from src.ml.model_store import get_model_store

logger = logging.getLogger(__name__)


class RULPredictor:
    def __init__(
        self,
        machine_id: str,
        beta: float = 2.1,
        eta: float = 720.0,
        max_step_pct: float | None = None,
        alarm_threshold_hours: float = 100.0,
        alarm_hysteresis_hours: float = 5.0,
        mode: str = "single",
        target_endpoint_hours: float | None = None,
        n_ticks_to_endpoint: int = 10,
    ):
        if mode not in ("single", "voting"):
            raise ValueError(f"mode must be 'single' or 'voting', got {mode!r}")
        if n_ticks_to_endpoint < 1:
            raise ValueError(f"n_ticks_to_endpoint must be >= 1, got {n_ticks_to_endpoint}")
        self.machine_id = machine_id
        self.beta = beta
        self.eta = eta
        self.mode = mode
        self.model = None
        self.ensemble: object | None = None
        self.feature_names: list = []
        self.conformal = ConformalRUL(alpha=0.10)
        store = get_model_store()
        self._model_path = str(store / f"{machine_id}_rul.joblib")
        self._prev_deg_proxy: float | None = None
        self._prev_rul: float | None = None
        import random as _random
        self._rng = _random.Random(
            int.from_bytes(machine_id.encode("utf-8")[:8].ljust(8, b"\x00"), "big")
            if isinstance(machine_id, str) else int(machine_id)
        )
        if target_endpoint_hours is not None:
            if target_endpoint_hours <= 0.0:
                raise ValueError(f"target_endpoint_hours must be > 0, got {target_endpoint_hours}")
            initial_rul = float(eta)
            if target_endpoint_hours >= initial_rul:
                derived_step = 1.0
            else:
                ratio = target_endpoint_hours / initial_rul
                derived_step = 1.0 - ratio ** (1.0 / float(n_ticks_to_endpoint))
            max_step_pct = float(derived_step)
        if max_step_pct is None:
            max_step_pct = 0.15
        if not 0.0 < float(max_step_pct) <= 1.0:
            raise ValueError(f"max_step_pct must be in (0, 1], got {max_step_pct}")
        self._max_step_pct = float(max_step_pct)
        self._target_endpoint_hours: float | None = (
            float(target_endpoint_hours) if target_endpoint_hours is not None else None
        )
        self._n_ticks_to_endpoint: int = int(n_ticks_to_endpoint)
        self._last_smoothed_rul: float | None = None
        self._alarm_threshold_hours = float(alarm_threshold_hours)
        self._alarm_hysteresis_hours = float(alarm_hysteresis_hours)
        self._alarm_state: str = "ARMED"
        if os.path.exists(self._model_path):
            self._load_model()

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> None:
        from xgboost import XGBRegressor

        from src.ml.model_card import compute_rul_holdout_metrics

        if len(X) < 50:
            logger.warning(f"{self.machine_id}: insufficient RUL training data ({len(X)} rows, minimum 50)")
            return
        self.feature_names = feature_names

        # Compute held-out metrics BEFORE fitting the production model so
        # the evaluation fit is independent of the final artefact.
        holdout_metrics = compute_rul_holdout_metrics(X, y)
        if holdout_metrics["status"] == "measured":
            logger.info(
                "[ML] RUL holdout eval: %s  MAE=%.2f h  RMSE=%.2f h  (n_test=%d)",
                self.machine_id,
                holdout_metrics["holdout_mae_hours"],
                holdout_metrics["holdout_rmse_hours"],
                holdout_metrics["holdout_n_test"],
            )
        else:
            logger.info(
                "[ML] RUL holdout eval skipped for %s: %s",
                self.machine_id,
                holdout_metrics.get("reason"),
            )

        if self.mode == "single":
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                n_jobs=2,
            )
            self.model.fit(X, y)
            self.ensemble = None
        else:
            from src.ml.models.rul_voting_ensemble import build_default_rul_voting_ensemble
            self.ensemble = build_default_rul_voting_ensemble()
            self.ensemble.fit(X, y)
            self.model = next(est for name, est in self.ensemble.members if name == "XGBoost")
        self._save_model(len(X), holdout_metrics=holdout_metrics)
        logger.info(f"[ML] RUL {self.mode} trained: {self.machine_id} ({len(X)} samples, mode={self.mode})")

    def predict(
        self,
        features: dict,
        phase: str,
        emergency_stop_count: int = 0,
        degradation_level: float | None = None,
    ) -> dict | None:
        if phase == "HEALTHY":
            # A healthy machine still has a finite remaining life: invert the
            # Weibull CDF at its streamed degradation level so the dashboard
            # always shows an age-consistent RUL instead of a blank.
            if degradation_level is None:
                return None
            result = self._degradation_based_estimate(degradation_level)
            result["method"] = "weibull_age"
            result["fallback"] = True
            result["model_trained"] = self.model is not None
            return result
        if phase == "FAILED":
            self._last_smoothed_rul = 0.0
            # Confidence is capped below 1.0 everywhere: no model is ever
            # entitled to claim certainty about remaining life.
            return {
                "rul_hours": 0.0,
                "rul_low_ci": 0.0,
                "rul_high_ci": 0.0,
                "confidence": 0.99,
                "failure_prob_24h": 100.0,
                "survive_shift_pct": 0.0,
                "method": "failed_state",
                "fallback": True,
                "model_trained": self.model is not None,
            }
        if self.model is None:
            result = self._weibull_only_estimate(
                features, emergency_stop_count, degradation_level
            )
            result["method"] = "weibull_fallback"
            result["fallback"] = True
            result["model_trained"] = False
            return self._with_conformal_interval(result)
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0)
        if self.mode == "voting" and self.ensemble is not None and getattr(self.ensemble, "is_fitted", False):
            rul_raw = float(self.ensemble.predict(X)[0])
        else:
            rul_raw = float(self.model.predict(X)[0])
        rul_raw = max(0.0, rul_raw)
        if emergency_stop_count > 0:
            from src.data_generator.machines import MACHINE_CONFIGS
            config = MACHINE_CONFIGS.get(self.machine_id, {})
            penalty_cfg = config.get("startup_penalty")
            if penalty_cfg:
                penalty_pct = self._rng.gauss(
                    penalty_cfg["rul_penalty_mu"],
                    penalty_cfg["rul_penalty_sigma"],
                )
                penalty_pct = max(-0.30, min(0.0, penalty_pct))
                rul_raw = rul_raw * (1 + penalty_pct * emergency_stop_count)
        rul_smoothed = self._smooth_rul(rul_raw)
        # The model extrapolates badly outside its training distribution
        # (e.g. an injected fault on a young machine): cap by the physical
        # bound implied by the observed degradation level.
        if degradation_level is not None:
            bound = self.degradation_rul_bound(degradation_level)
            if rul_smoothed > bound:
                rul_smoothed = bound
                self._last_smoothed_rul = bound
        ci = self._weibull_confidence_interval(rul_smoothed)
        failure_prob_24h = stats.weibull_min.cdf(24, self.beta, scale=self.eta) * 100
        survive_shift_pct = (1 - stats.weibull_min.cdf(rul_smoothed * 0.5, self.beta, scale=self.eta)) * 100
        result = {
            "rul_hours": round(max(0.0, rul_smoothed), 2),
            "rul_low_ci": round(ci["p10"], 2),
            "rul_high_ci": round(ci["p90"], 2),
            "confidence": round(min(ci["confidence"], 0.99), 3),
            "failure_prob_24h": round(failure_prob_24h, 2),
            "survive_shift_pct": round(survive_shift_pct, 2),
            "method": "xgboost+ema",
            "fallback": False,
            "model_trained": True,
        }
        return self._with_conformal_interval(result)

    def update_conformal(self, predicted_hours: float, actual_hours: float) -> None:
        self.conformal.update(predicted_hours, actual_hours)

    def _smooth_rul(self, raw_rul: float) -> float:
        raw_rul = max(0.0, float(raw_rul))
        if self._last_smoothed_rul is None:
            new_smoothed = raw_rul
        else:
            last = self._last_smoothed_rul
            cap_floor = last * (1.0 - self._max_step_pct)
            new_smoothed = max(raw_rul, cap_floor)
        self._last_smoothed_rul = new_smoothed
        return new_smoothed

    def should_publish_alarm(self, rul_hours: float) -> bool:
        rul = float(rul_hours)
        if self._alarm_state == "ARMED":
            if rul < self._alarm_threshold_hours:
                self._alarm_state = "TRIPPED"
                return True
            return False
        if rul > self._alarm_threshold_hours + self._alarm_hysteresis_hours:
            self._alarm_state = "ARMED"
        return False

    def reset_smoothing_state(self) -> None:
        self._last_smoothed_rul = None
        self._alarm_state = "ARMED"

    def _with_conformal_interval(self, result: dict) -> dict:
        interval = self.conformal.predict_with_interval(float(result["rul_hours"]))
        result["rul_low_ci"] = round(interval["rul_low_ci"], 2)
        result["rul_high_ci"] = round(interval["rul_high_ci"], 2)
        result["coverage_guarantee"] = interval["coverage_guarantee"]
        result["conformal_calibration_size"] = interval["conformal_calibration_size"]
        result["method"] = f"{result['method']}+conformal"
        return result

    def _weibull_only_estimate(
        self,
        features: dict,
        emergency_stop_count: int,
        degradation_level: float | None = None,
    ) -> dict:
        if degradation_level is not None:
            return self._degradation_based_estimate(degradation_level)
        z_scores = []
        for key, val in features.items():
            if "z_score" in key and val is not None:
                z_scores.append(abs(float(val)))
        if z_scores:
            raw_deg = min(1.0, max(0.0, (sum(z_scores) / len(z_scores)) / 3.0))
        else:
            raw_deg = 0.5
        alpha = 0.3
        if self._prev_deg_proxy is not None:
            deg_proxy = alpha * raw_deg + (1 - alpha) * self._prev_deg_proxy
        else:
            deg_proxy = raw_deg
        self._prev_deg_proxy = deg_proxy
        t = deg_proxy * self.eta
        rul = max(0.0, self.eta - t)
        max_delta = 0.1 * self.eta
        if self._prev_rul is not None:
            if rul > self._prev_rul + max_delta:
                rul = self._prev_rul + max_delta
            elif rul < self._prev_rul - max_delta:
                rul = self._prev_rul - max_delta
        self._prev_rul = rul
        ci = self._weibull_confidence_interval(rul)
        failure_prob_24h = stats.weibull_min.cdf(24, self.beta, scale=self.eta) * 100
        survive_shift_pct = (1 - stats.weibull_min.cdf(rul * 0.5, self.beta, scale=self.eta)) * 100
        return {
            "rul_hours": round(rul, 2),
            "rul_low_ci": round(ci["p10"], 2),
            "rul_high_ci": round(ci["p90"], 2),
            "confidence": round(0.5, 3),
            "failure_prob_24h": round(failure_prob_24h, 2),
            "survive_shift_pct": round(survive_shift_pct, 2),
        }

    def degradation_rul_bound(self, degradation_level: float) -> float:
        """Physical upper bound on RUL at an observed degradation level.

        Inverts the Weibull CDF (failure declared at d=0.95): whatever the
        regression model believes, a machine at d=0.93 does not have 90
        hours left. Pure function — no smoothing/state side effects.
        """
        d = max(0.0, min(float(degradation_level), 0.9499))
        consumed_fraction = (np.log(1.0 - d) / np.log(0.05)) ** (1.0 / self.beta)
        return max(0.0, self.eta * (1.0 - consumed_fraction))

    def _degradation_based_estimate(self, degradation_level: float) -> dict:
        """Physics-informed fallback: invert the Weibull CDF at the current
        degradation level to recover the consumed life fraction.

        With d = 1 - exp(-(t/eta)^beta) and failure declared at d=0.95
        (t95), the remaining life is eta_h * (1 - (ln(1-d)/ln(0.05))^(1/beta)).
        Unlike the z-score proxy this tracks the machine's actual age, so the
        estimate falls to zero as the machine approaches failure.
        """
        rul = self.degradation_rul_bound(degradation_level)
        self._prev_rul = rul
        ci = self._weibull_confidence_interval(rul)
        failure_prob_24h = stats.weibull_min.cdf(24, self.beta, scale=self.eta) * 100
        survive_shift_pct = (1 - stats.weibull_min.cdf(rul * 0.5, self.beta, scale=self.eta)) * 100
        return {
            "rul_hours": round(rul, 2),
            "rul_low_ci": round(ci["p10"], 2),
            "rul_high_ci": round(ci["p90"], 2),
            "confidence": round(0.6, 3),
            "failure_prob_24h": round(failure_prob_24h, 2),
            "survive_shift_pct": round(survive_shift_pct, 2),
        }

    def _solve_current_age(self, rul_predicted: float) -> float:
        if rul_predicted <= 0.0:
            return self.eta * 100.0
        unconditional_median = self.eta * (np.log(2.0) ** (1.0 / self.beta))
        if rul_predicted >= unconditional_median:
            return 0.0
        low, high = 0.0, self.eta * 100.0
        for _ in range(100):
            mid = (low + high) / 2.0
            median = self.eta * ((mid / self.eta) ** self.beta + np.log(2.0)) ** (1.0 / self.beta) - mid
            if median > rul_predicted:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    def _weibull_confidence_interval(self, rul_predicted: float) -> dict:
        from src.data_generator.machines import MACHINE_CONFIGS
        config = MACHINE_CONFIGS.get(self.machine_id, {}).get("weibull", {})
        beta_std = config.get("beta_std", 0.15)
        eta_std = config.get("eta_std", 45.0)
        current_age = self._solve_current_age(rul_predicted)
        rul_samples = []
        for _ in range(1000):
            b = max(0.5, self._rng.gauss(self.beta, beta_std))
            e = max(10.0, self._rng.gauss(self.eta, eta_std))
            u = self._rng.random()
            arg = (current_age / e) ** b - np.log(u)
            if arg > 0:
                r = e * (arg ** (1.0 / b)) - current_age
                rul_samples.append(max(0.0, r))
            else:
                rul_samples.append(0.0)
        rul_samples.sort()
        p10 = rul_samples[int(0.10 * len(rul_samples))]
        p90 = rul_samples[int(0.90 * len(rul_samples))]
        interval_width = p90 - p10
        confidence = max(0.0, min(1.0, 1.0 - interval_width / (rul_predicted + 1.0)))
        return {"p10": p10, "p90": p90, "confidence": confidence}

    def _save_model(self, training_rows: int, *, holdout_metrics: dict | None = None):
        store = get_model_store()
        os.makedirs(store, exist_ok=True)
        ensemble_payload = None
        if self.ensemble is not None:
            try:
                ensemble_payload = {
                    "members": [name for name, _ in self.ensemble.members],
                    "weights": [float(w) for w in (self.ensemble.weights or [])],
                }
            except Exception as exc:
                logger.debug("Could not serialise ensemble payload: %s", exc)
        joblib.dump(
            {
                "model": self.model,
                "ensemble": ensemble_payload,
                "mode": self.mode,
                "feature_names": self.feature_names,
                "beta": self.beta,
                "eta": self.eta,
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
            model_kind="rul_predictor",
            machine_id=self.machine_id,
            feature_list=self.feature_names,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "beta": self.beta,
                "eta": self.eta,
                "mode": self.mode,
                "max_step_pct": self._max_step_pct,
                "target_endpoint_hours": self._target_endpoint_hours,
                "n_ticks_to_endpoint": self._n_ticks_to_endpoint,
            },
            training_rows=training_rows,
            metrics=holdout_metrics,
        )

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
            self.model = data["model"]
            self.feature_names = data.get("feature_names", [])
            self.beta = data.get("beta", self.beta)
            self.eta = data.get("eta", self.eta)
            self.mode = data.get("mode", self.mode)
            ensemble_meta = data.get("ensemble")
            if ensemble_meta and self.mode == "voting":
                logger.info(
                    "%s: ensemble metadata found in artefact (%s); will be re-fit on next train() call",
                    self.machine_id,
                    ensemble_meta,
                )
            card_path = self._model_path.replace(".joblib", ".model_card.json")
            if not os.path.exists(card_path):
                write_model_card(
                    self._model_path,
                    model_kind="rul_predictor",
                    machine_id=self.machine_id,
                    feature_list=self.feature_names,
                    hyperparameters={"beta": self.beta, "eta": self.eta},
                    training_rows=None,
                )
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Failed to load {self._model_path}: {e}")

    @property
    def model_path(self) -> str:
        return self._model_path

    def _generate_rul_training_data(self, machine_id: str, db) -> tuple[np.ndarray | None, np.ndarray | None]:
        from datetime import datetime, timedelta

        import polars as pl

        from src.data_generator.machines import MACHINE_CONFIGS
        from src.database.models import SensorReading
        from src.ml.feature_engineering import compute_features
        failed_readings = (
            db.query(SensorReading)
            .filter(
                SensorReading.machine_id == machine_id,
                SensorReading.machine_phase == "FAILED",
            )
            .order_by(SensorReading.timestamp.desc())
            .limit(10)
            .all()
        )
        if len(failed_readings) < 1:
            logger.info(f"{machine_id}: No FAILED events found for RUL training")
            return None, None
        X_samples = []
        y_samples = []
        for failed_reading in failed_readings:
            failure_time = failed_reading.timestamp
            lookback_start = failure_time - timedelta(hours=24)
            readings = (
                db.query(SensorReading)
                .filter(
                    SensorReading.machine_id == machine_id,
                    SensorReading.timestamp >= lookback_start,
                    SensorReading.timestamp <= failure_time,
                    SensorReading.upstream_effect == False,
                )
                .order_by(SensorReading.timestamp.asc())
                .all()
            )
            if len(readings) < 50:
                continue
            df = pl.DataFrame({
                "timestamp": [r.timestamp.isoformat() if r.timestamp else "" for r in readings],
                "sensor_name": [r.sensor_name for r in readings],
                "value": [float(r.value) for r in readings],
                "machine_phase": [r.machine_phase or "HEALTHY" for r in readings],
                "upstream_effect": [r.upstream_effect or False for r in readings],
            })
            features_df = compute_features(df, MACHINE_CONFIGS.get(machine_id))
            if len(features_df) == 0:
                continue
            if not self.feature_names:
                self.feature_names = [k for k in features_df.row(0, named=True).keys() if k != "timestamp"]
            for i in range(len(features_df)):
                row = features_df.row(i, named=True)
                ts_str = row.get("timestamp", "")
                if not ts_str:
                    continue
                try:
                    reading_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    rul_hours = (failure_time - reading_time).total_seconds() / 3600
                    rul_hours = max(0.0, rul_hours)
                    feature_vec = [row.get(f, 0.0) for f in self.feature_names]
                    X_samples.append(feature_vec)
                    y_samples.append(rul_hours)
                except Exception as e:
                    logger.debug(f"Timestamp parse error: {e}")
                    continue
        if len(X_samples) < 50:
            logger.info(f"{machine_id}: Insufficient RUL training samples ({len(X_samples)})")
            return None, None
        X = np.array(X_samples)
        y = np.array(y_samples)
        X = np.nan_to_num(X, nan=0.0)
        logger.info(f"{machine_id}: Generated {len(X)} RUL training samples")
        return X, y
