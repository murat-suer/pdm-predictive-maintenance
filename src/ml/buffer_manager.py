from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime, timedelta

import polars as pl

from src.data_generator.machines import MACHINE_CONFIGS
from src.ml.feature_engineering import compute_features

logger = logging.getLogger(__name__)

PREDICTION_WARMUP_MIN = 96


class BufferManager(dict):
    """Dict subclass for type-safe buffer management."""

    def __init__(self, machine_configs=None):
        super().__init__()
        configs = machine_configs or MACHINE_CONFIGS
        for mid in configs:
            self[mid] = []

    def append(self, machine_id: str, payload: dict, max_size: int = 500) -> None:
        if machine_id not in self:
            return
        self[machine_id].append(payload)
        if len(self[machine_id]) > max_size:
            self[machine_id] = self[machine_id][-max_size:]

    def get_recent(self, machine_id: str, count: int = 30) -> list:
        """Son N okumayı döndür."""
        buffer = self.get(machine_id, [])
        return buffer[-count:] if buffer else []

    def set_buffer(self, machine_id: str, data: list) -> None:
        """Buffer'ı toplu olarak ayarla."""
        if machine_id in self:
            self[machine_id] = data[-500:]

    def size(self, machine_id: str) -> int:
        return len(self.get(machine_id, []))

    def all_ready(self, min_size: int) -> bool:
        return all(len(buf) >= min_size for buf in self.values())


def extract_latest_features(buffer: list, machine_profile: dict | None = None) -> dict:
    try:
        df = pl.DataFrame({
            "timestamp": [b.get("timestamp", "") for b in buffer],
            "sensor_name": [b.get("sensor_name", "") for b in buffer],
            "value": [float(b.get("value", 0)) for b in buffer],
            "machine_phase": [b.get("phase", "HEALTHY") for b in buffer],
            "upstream_effect": [b.get("upstream_effect", False) for b in buffer],
        })
        features_df = compute_features(df, machine_profile)
        if len(features_df) == 0:
            return {}
        last_row = features_df.row(-1, named=True)
        return {k: v for k, v in last_row.items() if k != "timestamp"}
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return {}


def generate_synthetic_calibration_buffer(machine_id: str, cycles: int = 120, seed: int = 42) -> list:
    rng = random.Random(seed)
    cfg = MACHINE_CONFIGS[machine_id]
    sensors = cfg["sensors"]
    buf = []
    base_time = datetime.now(UTC) - timedelta(seconds=cycles * 10)
    for cycle in range(cycles):
        ts = base_time + timedelta(seconds=cycle * 10)
        for sname, scfg in sensors.items():
            mu = scfg.get("nominal_mu", scfg.get("mu", 0))
            sigma = scfg.get("nominal_sigma", scfg.get("sigma", mu * 0.05))
            buf.append({
                "timestamp": ts.isoformat(),
                "machine_id": machine_id,
                "sensor_name": sname,
                "value": rng.gauss(mu, sigma),
                "machine_phase": "HEALTHY",
                "upstream_effect": False,
            })
    return buf


def warm_loaded_model_buffers(buffers: dict):
    warmed = []
    for machine_id, buffer in buffers.items():
        if len(buffer) >= PREDICTION_WARMUP_MIN:
            continue
        nominal_history = generate_synthetic_calibration_buffer(machine_id)
        buffers[machine_id] = (nominal_history + buffer)[-500:]
        warmed.append(machine_id)
    if warmed:
        logger.info(f"Seeded cold loaded-model buffers with nominal history: {', '.join(warmed)}")


def prefill_buffers_from_stream(r, buffers: dict, stream_key: str, max_events: int = 12000) -> int:
    try:
        events = r.xrevrange(stream_key, max="+", min="-", count=max_events)
    except Exception as exc:
        logger.warning(f"Redis stream prefill failed: {exc}")
        return 0
    count = 0
    for _event_id, fields in reversed(events):
        try:
            payload_raw = fields.get(b"payload", b"{}")
            payload = json.loads(payload_raw) if isinstance(payload_raw, bytes) else payload_raw
        except Exception:
            continue
        machine_id = payload.get("machine_id", "")
        if machine_id not in buffers:
            continue
        if payload.get("upstream_effect", False):
            continue
        phase = payload.get("phase", "HEALTHY")
        if phase in ("ANOMALY", "FAILED", "IDLE", "CALIBRATING", "CANARY_EVENT"):
            continue
        ts_raw = fields.get(b"timestamp", b"")
        payload["timestamp"] = ts_raw.decode() if isinstance(ts_raw, bytes) else str(ts_raw)
        buffers[machine_id].append(payload)
        if len(buffers[machine_id]) > 500:
            buffers[machine_id] = buffers[machine_id][-500:]
        count += 1
    return count
