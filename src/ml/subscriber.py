from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    force=True,
)

from src.ml.buffer_manager import (
    prefill_buffers_from_stream,
    warm_loaded_model_buffers,
)
from src.ml.calibration import (
    CALIBRATION_MIN,
    PipelineState,
    all_models_trained,
    build_detectors,
    build_predictors,
    train_all_models,
)
from src.ml.pipeline import (
    CONSUMER_GROUP,
    CONSUMER_NAME,
    STREAM_KEY,
    process_event,
)

logger = logging.getLogger(__name__)

pipeline_state = PipelineState()


def _ensure_consumer_group(r):
    import redis as redis_lib
    try:
        r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info(f"Consumer group '{CONSUMER_GROUP}' created")
    except redis_lib.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            pass
        else:
            raise


def _control_stream_tail(r) -> str:
    """Current tail of control_stream, so we only see future commands."""
    from src.data_generator.control import CONTROL_STREAM

    try:
        entries = r.xrevrange(CONTROL_STREAM, count=1)
        if entries:
            entry_id = entries[0][0]
            return entry_id.decode() if isinstance(entry_id, bytes) else entry_id
    except Exception:
        pass
    return "0-0"


def _check_maintenance_done(r, last_id: str, buffers: dict, machine_states: dict) -> str:
    """Flush a machine's rolling state after maintenance restored it.

    Stale degraded readings in the buffer would otherwise contaminate the
    rolling features (and re-alarm) right after the reset.
    """
    from src.data_generator.control import CONTROL_STREAM

    try:
        # No block parameter — non-blocking read.
        messages = r.xread({CONTROL_STREAM: last_id}, count=20)
    except Exception:
        return last_id
    for _stream, events in messages or []:
        for event_id, fields in events:
            last_id = event_id
            command = fields.get(b"command", fields.get("command", b""))
            command = command.decode() if isinstance(command, bytes) else command
            if command != "MAINTENANCE_DONE":
                continue
            machine_id = fields.get(b"machine_id", fields.get("machine_id", b""))
            machine_id = machine_id.decode() if isinstance(machine_id, bytes) else machine_id
            if machine_id in buffers:
                buffers[machine_id] = []
            if machine_id in machine_states:
                machine_states[machine_id] = {"emergency_stop_count": 0}
            logger.info(f"Maintenance done: cleared rolling state for {machine_id}")
    return last_id


def run():
    import redis as redis_lib

    from src.config import settings
    r = redis_lib.Redis.from_url(settings.REDIS_URL)
    _ensure_consumer_group(r)
    from src.ml.fault_classifier import FaultClassifier
    from src.ml.mhi_calculator import MHICalculator

    detectors = build_detectors()
    predictors = build_predictors()
    buffers = {mid: [] for mid in build_detectors()}
    machine_states = {mid: {"emergency_stop_count": 0} for mid in build_detectors()}
    mhi_calc = MHICalculator()
    fault_classifier = FaultClassifier()
    total_processed = prefill_buffers_from_stream(r, buffers, STREAM_KEY, max_events=12000)
    if total_processed:
        logger.info(f"Prefilled ML buffers: {total_processed} events")
    if all_models_trained(detectors):
        pipeline_state.set_production()
        warm_loaded_model_buffers(buffers)
        logger.info("Pre-trained models loaded — starting in PRODUCTION mode")
    if not pipeline_state.is_production():
        train_all_models(detectors, predictors, buffers)
        if all_models_trained(detectors):
            pipeline_state.set_production()
            logger.info("ML service switched to PRODUCTION mode from cold-start calibration")
    logger.info("ML subscriber started — waiting for sensor_data_stream events")
    control_last_id = _control_stream_tail(r)
    while True:
        control_last_id = _check_maintenance_done(
            r, control_last_id, buffers, machine_states
        )
        try:
            messages = r.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME, {STREAM_KEY: ">"}, count=200, block=2000,
            )
        except Exception as e:
            logger.error(f"Redis read error: {e}")
            continue
        if not messages:
            continue
        for _stream_name, events in messages:
            for event_id, fields in events:
                try:
                    process_event(
                        event_id, fields, detectors, predictors,
                        mhi_calc, buffers, r, machine_states, pipeline_state,
                        fault_classifier=fault_classifier,
                    )
                    r.xack(STREAM_KEY, CONSUMER_GROUP, event_id)
                except Exception as e:
                    logger.error(f"Event processing failed ({event_id}): {e}")
        if not pipeline_state.is_production():
            total_processed += len(events) if messages else 0
            if total_processed >= CALIBRATION_MIN:
                train_all_models(detectors, predictors, buffers)
                if all_models_trained(detectors):
                    pipeline_state.set_production()
                    logger.info("ML service switched to PRODUCTION mode")


if __name__ == "__main__":
    run()
