import json
from unittest.mock import MagicMock


def test_process_event_waits_for_prediction_warmup(monkeypatch):
    from src.ml import pipeline
    from src.ml.calibration import PipelineState
    state = PipelineState()
    state.set_production()
    detector = MagicMock()
    detector.model = object()
    fields = {
        b"timestamp": b"2026-05-21T12:00:00+00:00",
        b"payload": json.dumps({
            "machine_id": "AC-201",
            "sensor_name": "vibration_rms",
            "value": 1.2,
            "phase": "HEALTHY",
        }).encode(),
    }
    buffers = {"AC-201": []}
    pipeline.process_event(
        event_id=b"1-0",
        fields=fields,
        detectors={"AC-201": detector},
        predictors={},
        mhi_calc=MagicMock(),
        buffers=buffers,
        r=MagicMock(),
        machine_states={"AC-201": {"emergency_stop_count": 0}},
        pipeline_state=state,
    )
    assert len(buffers["AC-201"]) == 1
    detector.predict.assert_not_called()


def test_warm_loaded_model_buffers_seeds_short_machine_history(monkeypatch):
    from src.ml import buffer_manager
    monkeypatch.setattr(
        buffer_manager,
        "generate_synthetic_calibration_buffer",
        lambda machine_id, **kw: [{"machine_id": machine_id}] * 120,
    )
    buffers = {
        "AC-201": [{"machine_id": "AC-201"}],
        "HX-202": [{"machine_id": "HX-202"}] * buffer_manager.PREDICTION_WARMUP_MIN,
    }
    buffer_manager.warm_loaded_model_buffers(buffers)
    assert len(buffers["AC-201"]) == 121
    assert len(buffers["HX-202"]) == buffer_manager.PREDICTION_WARMUP_MIN


def test_anomaly_confirmation_requires_two_recent_cycles():
    from src.ml.pipeline import anomaly_is_confirmed
    state = {}
    assert anomaly_is_confirmed(state, "2026-05-21T12:00:00+00:00") is False
    assert anomaly_is_confirmed(state, "2026-05-21T12:00:00+00:00") is False
    assert anomaly_is_confirmed(state, "2026-05-21T12:00:10+00:00") is True


def test_anomaly_confirmation_expires_old_cycle_evidence():
    from src.ml.pipeline import anomaly_is_confirmed
    state = {}
    assert anomaly_is_confirmed(state, "2026-05-21T12:00:00+00:00") is False
    assert anomaly_is_confirmed(state, "2026-05-21T12:02:00+00:00") is False


def test_prediction_grace_counts_unique_live_cycles():
    from src.ml.pipeline import PREDICTION_GRACE_CYCLES, in_prediction_grace
    state = {}
    for second in range(PREDICTION_GRACE_CYCLES):
        timestamp = f"2026-05-21T12:00:{second * 10:02d}+00:00"
        assert in_prediction_grace(state, timestamp) is True
        assert in_prediction_grace(state, timestamp) is True
    assert in_prediction_grace(state, "2026-05-21T12:01:00+00:00") is False


def test_synthetic_calibration_uses_one_timestamp_per_sensor_cycle():
    from src.data_generator.machines import MACHINE_CONFIGS
    from src.ml.buffer_manager import generate_synthetic_calibration_buffer
    buffer = generate_synthetic_calibration_buffer("AC-201", cycles=2)
    sensor_count = len(MACHINE_CONFIGS["AC-201"]["sensors"])
    assert len({row["timestamp"] for row in buffer[:sensor_count]}) == 1
    assert len({row["timestamp"] for row in buffer[sensor_count:]}) == 1
    assert buffer[0]["timestamp"] != buffer[sensor_count]["timestamp"]


def test_low_score_healthy_outlier_does_not_enter_alarm_workflow():
    from src.ml.pipeline import is_alarm_candidate
    assert is_alarm_candidate("HEALTHY", True, 0.63) is False
    assert is_alarm_candidate("HEALTHY", True, 0.91) is True
    assert is_alarm_candidate("ANOMALY", True, 0.63) is True


def test_publish_confirmed_fault_event_serializes_payload():
    from src.ml.event_publisher import SYSTEM_STREAM, publish_confirmed_fault_event
    from src.ml.fault_aggregator import ConfirmedFault
    redis_client = MagicMock()
    publish_confirmed_fault_event(
        redis_client,
        ConfirmedFault(
            machine_id="AC-201",
            fault_type="BEARING_FAULT",
            confidence=0.72,
            votes=4,
            window_size=5,
            time_to_consensus_s=20.0,
        ),
    )
    stream, fields = redis_client.xadd.call_args.args
    assert stream == SYSTEM_STREAM
    assert fields["type"] == "confirmed_fault"
    assert json.loads(fields["payload"])["votes"] == 4


def test_publish_diagnosis_event_serializes_payload():
    from src.ml.event_publisher import SYSTEM_STREAM, publish_diagnosis_event
    from src.ml.fault_narrator import FaultDiagnosis
    redis_client = MagicMock()
    publish_diagnosis_event(
        redis_client,
        FaultDiagnosis(machine_id="AC-201", fault_type="BEARING_FAULT", confidence=0.8),
    )
    stream, fields = redis_client.xadd.call_args.args
    assert stream == SYSTEM_STREAM
    assert fields["type"] == "diagnosis"
    assert json.loads(fields["payload"])["fault_type"] == "BEARING_FAULT"


def test_failed_event_updates_conformal_from_latest_rul_prediction():
    from src.ml.pipeline import _update_conformal_from_failure, remember_rul_prediction
    predictor = MagicMock()
    machine_state = {}
    remember_rul_prediction(
        machine_state,
        {"rul_hours": 12.0},
        "2026-05-21T12:00:00+00:00",
    )
    _update_conformal_from_failure(
        predictor,
        machine_state,
        "2026-05-21T15:30:00+00:00",
    )
    predictor.update_conformal.assert_called_once_with(12.0, 3.5)
    assert "latest_rul_prediction" not in machine_state


def test_publish_early_warning_event_serializes_sensor_score():
    from src.ml.event_publisher import SYSTEM_STREAM, publish_early_warning_event
    redis_client = MagicMock()
    publish_early_warning_event(
        redis_client,
        "AC-201",
        "vibration_rms",
        5.75,
        "2026-05-21T12:00:00+00:00",
    )
    stream, fields = redis_client.xadd.call_args.args
    assert stream == SYSTEM_STREAM
    assert fields["type"] == "early_warning"
    assert fields["sensor"] == "vibration_rms"
    assert fields["cusum_score"] == "5.75"
