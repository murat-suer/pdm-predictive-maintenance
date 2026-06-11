import logging
import random
from datetime import UTC, datetime, timedelta
from uuid import uuid4

logger = logging.getLogger(__name__)


class CanaryProbeSystem:
    _probe_history: list = []

    PROBE_SCENARIOS = {
        "AC": [
            {"name": "bearing_fault_stage2", "vibration_rms_mult": 2.5, "bearing_temp_delta": 15},
            {"name": "oil_pressure_low", "oil_pressure_mult": 0.4, "motor_current_mult": 1.3},
        ],
        "HX": [
            {"name": "fouling_high", "fouling_index_mult": 5.0, "pressure_drop_mult": 2.0},
            {"name": "flow_blockage", "flow_rate_mult": 0.3, "pressure_drop_mult": 1.8},
        ],
        "CM": [
            {"name": "belt_tension_high", "belt_tension_mult": 1.8, "drive_temp_delta": 20},
            {"name": "motor_overload", "motor_load_delta": +25, "speed_rpm_mult": 0.92},
        ],
    }

    def run_probe(
        self,
        machine_id: str,
        machine_type: str,
        anomaly_detector,
        triggered_by: str = "SCHEDULED",
    ) -> dict:
        probe_id = str(uuid4())
        scenario = random.choice(self.PROBE_SCENARIOS.get(machine_type, [{}]))
        started_at = datetime.now(UTC)
        logger.info(f"Canary probe [{probe_id}]: {machine_id} scenario={scenario.get('name')}")
        synthetic_features = self._build_synthetic_features(machine_id, machine_type, scenario, anomaly_detector)
        result = anomaly_detector.predict(synthetic_features)
        detected = result.get("is_anomaly", False)
        completed_at = datetime.now(UTC)
        duration_s = int((completed_at - started_at).total_seconds())
        success = detected
        log_entry = {
            "probe_id": probe_id,
            "machine_id": machine_id,
            "probe_type": machine_type,
            "scenario": scenario.get("name"),
            "started_at": started_at,
            "completed_at": completed_at,
            "detected": detected,
            "expected": True,
            "success": success,
            "duration_s": duration_s,
            "triggered_by": triggered_by,
            "recalibration_triggered": not success,
            "notes": f"Score: {result.get('anomaly_score', 0):.3f}" if result else None,
        }
        self._probe_history.append(log_entry)
        if len(self._probe_history) > 500:
            self._probe_history = self._probe_history[-500:]
        if not success:
            logger.warning(f"Canary probe FAILED for {machine_id}: model did not detect synthetic anomaly")
        return log_entry

    @classmethod
    def get_model_metrics(cls, db_session=None) -> dict:
        anomaly_rate = 0.0
        total = 0
        anomaly_count = 0
        if db_session is not None:
            try:
                from sqlalchemy import func

                from src.database.models import SensorReading
                window_start = datetime.now(UTC) - timedelta(hours=24)
                total = (
                    db_session.query(func.count())
                    .filter(
                        SensorReading.timestamp > window_start,
                        SensorReading.upstream_effect == False,
                    )
                    .scalar()
                    or 0
                )
                if total > 0:
                    anomaly_count = (
                        db_session.query(func.count())
                        .filter(
                            SensorReading.timestamp > window_start,
                            SensorReading.is_anomaly == True,
                            SensorReading.upstream_effect == False,
                        )
                        .scalar()
                        or 0
                    )
                    anomaly_rate = round(anomaly_count / total, 4)
            except Exception:
                pass
        total_probes = len(cls._probe_history)
        detected = sum(1 for p in cls._probe_history if p.get("detected", False))
        missed = total_probes - detected
        detection_rate = round(detected / total_probes, 4) if total_probes > 0 else None
        last_probe = cls._probe_history[-1] if cls._probe_history else None
        return {
            "anomaly_rate": anomaly_rate,
            "anomaly_count": anomaly_count,
            "total_readings": total,
            "canary_detection_rate": detection_rate,
            "total_canary_probes": total_probes,
            "canary_detected": detected,
            "canary_missed": missed,
            "last_canary_result": last_probe,
        }

    def _build_synthetic_features(self, machine_id, machine_type, scenario, anomaly_detector):
        from src.data_generator.machines import MACHINE_CONFIGS
        config = MACHINE_CONFIGS.get(machine_id, {})
        sensors = config.get("sensors", {})
        features = {}
        for sensor_name, s_cfg in sensors.items():
            mu = s_cfg["nominal_mu"]
            sigma = s_cfg.get("nominal_sigma", mu * 0.05)
            mult_key = f"{sensor_name}_mult"
            delta_key = f"{sensor_name}_delta"
            is_affected = mult_key in scenario or delta_key in scenario
            if mult_key in scenario:
                value = mu * scenario[mult_key]
            elif delta_key in scenario:
                value = mu + scenario[delta_key]
            else:
                value = mu
            if anomaly_detector.feature_names:
                for feat in anomaly_detector.feature_names:
                    if not feat.startswith(sensor_name):
                        continue
                    if feat.endswith("_value"):
                        features[feat] = value
                    elif feat.endswith("_rolling_mean_5m"):
                        features[feat] = mu + 0.5 * sigma if is_affected else mu
                    elif feat.endswith("_rolling_std_5m"):
                        features[feat] = 2.0 * sigma if is_affected else sigma
                    elif feat.endswith("_rate_of_change"):
                        features[feat] = value - mu if is_affected else 0.0
                    elif feat.endswith("_z_score") or feat.endswith("_shift_adj_z"):
                        features[feat] = 5.0 if (is_affected and feat.endswith("_shift_adj_z")) else (4.0 if is_affected else 0.0)
                    else:
                        features[feat] = 0.0
        return features


class DriftDetector:
    ANOMALY_RATE_WINDOW_HOURS = 24
    ANOMALY_RATE_THRESHOLD = 0.15
    ALIGNMENT_THRESHOLD = 0.30
    FP_RATE_THRESHOLD = 0.20

    def check_anomaly_rate_drift(self, machine_id: str, db_session) -> dict:
        from sqlalchemy import func

        from src.database.models import SensorReading
        window_start = datetime.now(UTC) - timedelta(hours=self.ANOMALY_RATE_WINDOW_HOURS)
        total = (
            db_session.query(func.count())
            .filter(
                SensorReading.machine_id == machine_id,
                SensorReading.timestamp > window_start,
            )
            .scalar()
            or 1
        )
        anomaly_count = (
            db_session.query(func.count())
            .filter(
                SensorReading.machine_id == machine_id,
                SensorReading.timestamp > window_start,
                SensorReading.is_anomaly == True,
                SensorReading.upstream_effect == False,
            )
            .scalar()
            or 0
        )
        rate = anomaly_count / total
        drift = rate > self.ANOMALY_RATE_THRESHOLD
        return {
            "drift_detected": drift,
            "anomaly_rate": round(rate, 4),
            "anomaly_count": anomaly_count,
            "total_readings": total,
            "message": f"Anomaly rate: {rate:.1%}" + (" — DRIFT ALERT" if drift else ""),
        }

    def check_operator_alignment(self, machine_id: str, db_session) -> dict:
        from src.database.models import DecisionAuditLog
        window_start = datetime.now(UTC) - timedelta(days=7)
        decisions = (
            db_session.query(DecisionAuditLog)
            .filter(
                DecisionAuditLog.alarm_id != None,
                DecisionAuditLog.created_at > window_start,
            )
            .all()
        )
        if not decisions:
            return {
                "drift_detected": False,
                "operator_alignment_pct": 100.0,
                "override_count": 0,
                "total_decisions": 0,
                "message": "No decisions in window",
            }
        total = len(decisions)
        override = sum(1 for d in decisions if d.overridden)
        alignment_pct = round((1 - override / total) * 100, 1)
        drift = (override / total) > self.ALIGNMENT_THRESHOLD
        return {
            "drift_detected": drift,
            "operator_alignment_pct": alignment_pct,
            "override_count": override,
            "total_decisions": total,
            "message": f"Alignment: {alignment_pct}%" + (" — DRIFT ALERT" if drift else ""),
        }

    def check_maintenance_validation(self, machine_id: str, db_session) -> dict:
        from src.database.models import MaintenanceLog
        window_start = datetime.now(UTC) - timedelta(days=30)
        logs = (
            db_session.query(MaintenanceLog)
            .filter(
                MaintenanceLog.machine_id == machine_id,
                MaintenanceLog.performed_at != None,
                MaintenanceLog.fault_found != None,
                MaintenanceLog.performed_at > window_start,
            )
            .all()
        )
        if not logs:
            return {
                "drift_detected": False,
                "false_positive_rate": 0.0,
                "fp_count": 0,
                "total_maintenance": 0,
                "message": "No completed maintenance in window",
            }
        total = len(logs)
        fps = sum(1 for log in logs if log.fault_found is False)
        fp_rate = fps / total
        drift = fp_rate > self.FP_RATE_THRESHOLD
        return {
            "drift_detected": drift,
            "false_positive_rate": round(fp_rate, 4),
            "fp_count": fps,
            "total_maintenance": total,
            "message": f"FP rate: {fp_rate:.1%}" + (" — DRIFT ALERT" if drift else ""),
        }

    def run_all_checks(self, machine_id: str, db_session) -> dict:
        rate_result = self.check_anomaly_rate_drift(machine_id, db_session)
        align_result = self.check_operator_alignment(machine_id, db_session)
        fp_result = self.check_maintenance_validation(machine_id, db_session)
        any_drift = rate_result["drift_detected"] or align_result["drift_detected"] or fp_result["drift_detected"]
        return {
            "machine_id": machine_id,
            "checked_at": datetime.now(UTC).isoformat(),
            "drift_detected": any_drift,
            "anomaly_rate": rate_result,
            "alignment": align_result,
            "fp_validation": fp_result,
        }
