"""Sliding-window voting over fault classifications."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass

import redis

WINDOW_SIZE = 5
MIN_AGREEMENT = 3
WINDOW_TTL_SECONDS = 60


@dataclass(frozen=True)
class ConfirmedFault:
    """Fault classification that reached temporal consensus."""

    machine_id: str
    fault_type: str
    confidence: float
    votes: int
    window_size: int
    time_to_consensus_s: float


class FaultAggregator:
    """Confirm repeated fault classifications in a Redis-backed window."""

    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client

    def submit(
        self,
        machine_id: str,
        fault_type: str | None,
        confidence: float,
    ) -> ConfirmedFault | None:
        """Record one classification and return consensus when it exists."""
        if fault_type is None:
            return None

        entry = json.dumps({"t": time.time(), "ft": fault_type, "c": confidence})
        key = self._key(machine_id)
        pipe = self.r.pipeline()
        pipe.rpush(key, entry)
        pipe.ltrim(key, -WINDOW_SIZE, -1)
        pipe.expire(key, WINDOW_TTL_SECONDS)
        pipe.lrange(key, 0, -1)
        _, _, _, window = pipe.execute()

        items = [_decode_entry(item) for item in window]
        top_fault, votes = Counter(item["ft"] for item in items).most_common(1)[0]
        if votes < MIN_AGREEMENT or top_fault != fault_type:
            return None

        matching = [item for item in items if item["ft"] == fault_type]
        mean_confidence = sum(item["c"] for item in matching) / len(matching)
        return ConfirmedFault(
            machine_id=machine_id,
            fault_type=fault_type,
            confidence=(votes / WINDOW_SIZE) * mean_confidence,
            votes=votes,
            window_size=WINDOW_SIZE,
            time_to_consensus_s=items[-1]["t"] - items[0]["t"],
        )

    @staticmethod
    def _key(machine_id: str) -> str:
        return f"fault_window:{machine_id}"


def _decode_entry(entry: bytes | str) -> dict:
    """Decode Redis text/bytes entries to a vote payload."""
    if isinstance(entry, bytes):
        entry = entry.decode()
    return json.loads(entry)
