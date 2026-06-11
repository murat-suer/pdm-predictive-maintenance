"""WebSocket live-data layer.

Broadcasts two kinds of messages to connected dashboard clients:
- ``snapshot``: periodic fleet snapshot built from the database (every 5 s)
- ``anomaly``: anomaly events forwarded from the Redis ``anomaly_stream``

Redis being unavailable degrades gracefully to snapshots only.
"""
import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from src.config import settings

logger = logging.getLogger(__name__)

SNAPSHOT_INTERVAL_SECONDS = 5
ANOMALY_STREAM = "anomaly_stream"


class ConnectionManager:
    """Tracks open WebSocket connections and broadcasts JSON messages."""

    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._connections:
            return
        payload = json.dumps(message, default=str)
        async with self._lock:
            connections = list(self._connections)
        dead: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for connection in dead:
            await self.disconnect(connection)


manager = ConnectionManager()


def build_snapshot() -> dict[str, Any]:
    """Build the periodic fleet snapshot (runs in a worker thread)."""
    from src.api.routers.common import (
        derive_machine_status,
        latest_health_scores,
    )
    from src.data_generator.machines import MACHINE_CONFIGS
    from src.database.connection import get_db_context
    from src.database.models import DecisionLog

    machines: list[dict[str, Any]] = []
    pending_decisions = 0
    with get_db_context() as db:
        machine_ids = list(MACHINE_CONFIGS.keys())
        scores = latest_health_scores(db, machine_ids)
        for machine_id in machine_ids:
            status, top_alarm = derive_machine_status(db, machine_id)
            score = scores.get(machine_id)
            machines.append(
                {
                    "id": machine_id,
                    "status": status,
                    "top_alarm": top_alarm,
                    "health_score": score.health_score if score else None,
                    "rul_hours": score.rul_hours if score else None,
                    "reliability": round(score.reliability_score * 100.0, 1)
                    if score and score.reliability_score is not None
                    else None,
                }
            )
        pending_decisions = (
            db.query(DecisionLog).filter(DecisionLog.action == "PENDING").count()
        )

    return {"type": "snapshot", "machines": machines, "pending_decisions": pending_decisions}


async def snapshot_loop() -> None:
    """Push a fleet snapshot to all clients on a fixed interval."""
    while True:
        try:
            if manager.active_count > 0:
                snapshot = await asyncio.to_thread(build_snapshot)
                await manager.broadcast(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Snapshot broadcast failed: {e}")
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)


async def anomaly_forward_loop() -> None:
    """Forward anomaly_stream events to clients as they arrive."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.warning("redis package unavailable; anomaly forwarding disabled")
        return

    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    last_id = "$"
    while True:
        try:
            messages = await client.xread({ANOMALY_STREAM: last_id}, count=20, block=5000)
            for _stream, events in messages or []:
                for event_id, fields in events:
                    last_id = event_id
                    await manager.broadcast({"type": "anomaly", "event": fields})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Anomaly stream read failed: {e}; retrying in 5s")
            await asyncio.sleep(5)


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Per-client WebSocket handler: initial snapshot, then broadcast-driven."""
    await manager.connect(websocket)
    try:
        snapshot = await asyncio.to_thread(build_snapshot)
        await websocket.send_text(json.dumps(snapshot, default=str))
        while True:
            # Client messages are only keep-alives; ignore content.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)
