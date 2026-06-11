from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

ANOMALY_STREAM = "anomaly_stream"
SYSTEM_STREAM = "system_stream"
ANOMALY_CRITICAL_SCORE_MIN = 0.75


def publish_anomaly_event(r, machine_id, anomaly_id, det_result, payload, rul_hours=None):
    fields = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "pm-ml",
        "event_type": "ANOMALY_DETECTED",
        "machine_id": machine_id,
        "anomaly_id": str(anomaly_id),
        "anomaly_score": str(det_result["anomaly_score"]),
        "top_contributing_sensor": str(det_result.get("top_contributing_sensor", "")),
        "severity": "CRITICAL" if det_result["anomaly_score"] > ANOMALY_CRITICAL_SCORE_MIN else "WARNING",
        "phase": payload.get("phase", "HEALTHY"),
    }
    if rul_hours is not None:
        fields["rul_hours"] = str(rul_hours)
    r.xadd(ANOMALY_STREAM, fields, maxlen=10000)


def publish_confirmed_fault_event(r, confirmed_fault):
    r.xadd(
        SYSTEM_STREAM,
        {
            "type": "confirmed_fault",
            "payload": json.dumps(asdict(confirmed_fault)),
        },
        maxlen=5000,
    )


def publish_diagnosis_event(r, diagnosis):
    r.xadd(
        SYSTEM_STREAM,
        {
            "type": "diagnosis",
            "payload": json.dumps(diagnosis.to_dict()),
        },
        maxlen=5000,
    )


def publish_unknown_diagnosis_event(r, diagnosis, anomaly_id):
    try:
        r.xadd(
            SYSTEM_STREAM,
            {
                "type": "diagnosis_unknown",
                "machine_id": diagnosis.machine_id,
                "anomaly_id": str(anomaly_id) if anomaly_id is not None else "",
                "confidence": f"{diagnosis.confidence:.4f}",
                "recommended_action": "DISPATCH_TECHNICIAN",
                "narrative": (
                    "Sensor verilerinden anlamli bir sonuc cikarilamadi. "
                    "Teknisyen gonderin, tanilamayi yerinde yapsin."
                ),
                "ts": datetime.now(UTC).isoformat(),
            },
            maxlen=5000,
        )
    except Exception as exc:
        logger.warning(f"diagnosis_unknown publish failed: {exc}")


def publish_early_warning_event(r, machine_id, sensor_name, cusum_score, timestamp):
    r.xadd(
        SYSTEM_STREAM,
        {
            "type": "early_warning",
            "machine_id": machine_id,
            "sensor": sensor_name,
            "cusum_score": str(cusum_score),
            "timestamp": timestamp or datetime.now(UTC).isoformat(),
        },
        maxlen=5000,
    )


def publish_predicted_failure(
    r, machine_id, det_result, payload, rul_result, predictors,
    failure_cost_per_hour_fn=None,
):
    if not rul_result:
        return
    rul_hours_raw = rul_result.get("rul_hours")
    if rul_hours_raw is None:
        return
    try:
        rul_hours = float(rul_hours_raw)
    except (TypeError, ValueError):
        return
    predictor = predictors.get(machine_id)
    if predictor is None:
        return
    if not predictor.should_publish_alarm(rul_hours):
        return
    severity = "CRITICAL" if rul_hours < predictor._alarm_threshold_hours * 0.5 else "WARNING"
    top_sensor = det_result.get("top_contributing_sensor", "") or ""
    degradation_level = float(payload.get("degradation_level", 0.0) or 0.0)
    anomaly_score = float(det_result.get("anomaly_score", 0.0) or 0.0)
    try:
        anomaly_id = _write_rul_triggered_anomaly_log(
            machine_id=machine_id,
            anomaly_score=anomaly_score,
            severity=severity,
            top_sensor=top_sensor,
            rul_hours=rul_hours,
            degradation_level=degradation_level,
        )
    except Exception as exc:
        logger.error(f"Failed to insert RUL-triggered AnomalyLog for {machine_id}: {exc}")
        return
    r.xadd(
        ANOMALY_STREAM,
        {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "source": "pm-ml-rul",
            "event_type": "PREDICTED_FAILURE",
            "machine_id": machine_id,
            "anomaly_id": str(anomaly_id),
            "anomaly_score": str(anomaly_score),
            "top_contributing_sensor": str(top_sensor),
            "severity": severity,
            "phase": payload.get("phase", "HEALTHY"),
            "degradation_level": f"{degradation_level:.4f}",
            "rul_hours": f"{rul_hours:.2f}",
        },
        maxlen=10000,
    )
    logger.warning(
        f"RUL TRIGGER [{machine_id}] rul={rul_hours:.2f}h "
        f"< threshold={predictor._alarm_threshold_hours:.1f}h"
    )


def _write_rul_triggered_anomaly_log(machine_id, anomaly_score, severity, top_sensor, rul_hours, degradation_level):
    from src.database.connection import get_db_context
    from src.database.models import AnomalyLog
    with get_db_context() as db:
        row = AnomalyLog(
            machine_id=machine_id,
            detected_at=datetime.now(UTC),
            anomaly_score=float(anomaly_score),
            shap_values=None,
            top_contributing_sensor=top_sensor or None,
            severity=severity,
            status="ACTIVE",
            upstream_effect=False,
            fault_type="RUL_PREDICTED_FAILURE",
            fault_confidence=0.0,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
