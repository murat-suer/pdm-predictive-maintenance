from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select

from src.database.connection import get_db_context
from src.database.models import AnomalyLog, MachineHealthScore, SensorReading

logger = logging.getLogger(__name__)

ANOMALY_CRITICAL_SCORE_MIN = 0.75


def write_anomaly_log(
    machine_id: str,
    payload: dict,
    det_result: dict,
    features: dict = None,
    sensor_snapshot: dict = None,
    fault_classifier=None,
    compliance_logger=None,
) -> tuple:
    with get_db_context() as db:
        detected_at = datetime.now(UTC)
        severity = "CRITICAL" if det_result["anomaly_score"] > ANOMALY_CRITICAL_SCORE_MIN else "WARNING"
        shap_values = det_result.get("shap_values") or {}
        top_sensor = det_result.get("top_contributing_sensor")
        machine_type = machine_id.split("-")[0]
        sensor_readings = dict(sensor_snapshot or {})
        if payload.get("sensor_name") and payload.get("value") is not None:
            sensor_readings[payload["sensor_name"]] = float(payload["value"])
        fault_result = None
        if fault_classifier is not None:
            fault_result = fault_classifier.classify(
                machine_id=machine_id,
                machine_type=machine_type,
                anomaly_score=det_result["anomaly_score"],
                shap_values=shap_values,
                sensor_readings=sensor_readings,
                top_contributing_sensor=top_sensor,
            )
        existing = db.execute(
            select(AnomalyLog)
            .where(
                AnomalyLog.machine_id == machine_id,
                AnomalyLog.status == "ACTIVE",
            )
            .order_by(AnomalyLog.detected_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if existing is not None:
            if det_result["anomaly_score"] > existing.anomaly_score:
                existing.anomaly_score = det_result["anomaly_score"]
                existing.shap_values = shap_values
                existing.top_contributing_sensor = top_sensor
                existing.severity = severity
                if fault_result:
                    existing.fault_type = fault_result.fault_type
                    existing.fault_confidence = round(fault_result.fault_confidence, 3)
            db.commit()
            return existing.id, False, fault_result
        record = AnomalyLog(
            machine_id=machine_id,
            detected_at=detected_at,
            anomaly_score=det_result["anomaly_score"],
            shap_values=shap_values,
            top_contributing_sensor=top_sensor,
            severity=severity,
            status="ACTIVE",
            upstream_effect=False,
            fault_type=fault_result.fault_type if fault_result else None,
            fault_confidence=round(fault_result.fault_confidence, 3) if fault_result else None,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id, True, fault_result


def update_mhi(machine_id, phase, payload, det_result, rul_result, mhi_calc, buffers):
    deg = payload.get("degradation_level", 0.0)
    recent_anomaly_count = 0
    recent_readings_count = 0
    if machine_id in buffers:
        recent = buffers[machine_id][-30:]
        recent_anomaly_count = sum(1 for r in recent if r.get("is_anomaly"))
        recent_readings_count = len(recent)
    # Reliability is grounded in CONFIRMED anomaly events (what the alarm
    # pipeline actually raised in the last 24h), not raw detector flags —
    # a grumbling detector on a healthy machine must not read as 0%.
    confirmed_events = None
    try:
        from datetime import timedelta

        from src.database.models import MaintenanceLog

        with get_db_context() as db:
            # An overhaul renews the unit: events from before the last
            # repair belong to the previous life and must not depress the
            # rebuilt machine's reliability.
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            last_repair = (
                db.query(MaintenanceLog.performed_at)
                .filter(MaintenanceLog.machine_id == machine_id)
                .order_by(MaintenanceLog.performed_at.desc())
                .first()
            )
            if last_repair is not None and last_repair[0] is not None:
                repaired_at = last_repair[0]
                if repaired_at.tzinfo is None:
                    repaired_at = repaired_at.replace(tzinfo=UTC)
                cutoff = max(cutoff, repaired_at)
            confirmed_events = (
                db.query(AnomalyLog)
                .filter(
                    AnomalyLog.machine_id == machine_id,
                    AnomalyLog.detected_at >= cutoff,
                )
                .count()
            )
    except Exception as e:
        logger.warning(f"Confirmed-event count failed for {machine_id}: {e}")
    mhi = mhi_calc.compute(
        machine_id=machine_id,
        phase=phase,
        degradation_level=deg,
        recent_anomaly_count=recent_anomaly_count,
        recent_readings_count=recent_readings_count,
        confirmed_events=confirmed_events,
    )
    if rul_result:
        mhi["rul_hours"] = float(rul_result.get("rul_hours")) if rul_result.get("rul_hours") is not None else None
        mhi["confidence"] = float(rul_result.get("confidence")) if rul_result.get("confidence") is not None else None
    sanitized = {}
    for k, v in mhi.items():
        if hasattr(v, "item"):
            sanitized[k] = v.item()
        else:
            sanitized[k] = v
    try:
        with get_db_context() as db:
            db.add(
                MachineHealthScore(
                    **{k: v for k, v in sanitized.items() if k != "calculated_at"}
                    | {"calculated_at": sanitized["calculated_at"]}
                )
            )
            db.commit()
    except Exception as e:
        logger.error(f"MHI update failed for {machine_id}: {e}")


def update_sensor_reading_anomaly(machine_id: str, anomaly_score: float):
    try:
        with get_db_context() as db:
            stmt = (
                select(SensorReading)
                .where(
                    SensorReading.machine_id == machine_id,
                    SensorReading.is_anomaly == False,
                    SensorReading.upstream_effect == False,
                )
                .order_by(SensorReading.timestamp.desc())
                .limit(5)
            )
            for row in db.execute(stmt).scalars():
                row.is_anomaly = True
                row.anomaly_score = anomaly_score
            db.commit()
    except Exception as e:
        logger.error(f"Anomaly flag update failed: {e}")
