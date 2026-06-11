"""Production runner for the data-generator service.

Drives SimulationEngine over the six configured machines and ships every
reading to its two consumers:
- Redis ``sensor_data_stream`` (real-time path for the ML subscriber)
- TimescaleDB ``sensor_readings`` (historical path for RUL training & API)

Run with: ``python -m src.data_generator.runner``
"""
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

STREAM_KEY = "sensor_data_stream"
STREAM_MAXLEN = 50_000
DB_FLUSH_BATCH = 200


class ReadingSink:
    """Buffers readings and flushes them to Redis and TimescaleDB."""

    def __init__(self, redis_client, session_factory):
        self._redis = redis_client
        self._session_factory = session_factory
        self._db_batch: list[dict] = []

    async def __call__(self, readings) -> None:
        for reading in readings:
            payload = {
                "machine_id": reading.machine_id,
                "sensor_name": reading.sensor_name,
                "value": reading.value,
                "phase": reading.phase,
                "degradation_level": reading.degradation_level,
                "upstream_effect": False,
                "present": reading.present,
            }
            timestamp = datetime.fromtimestamp(reading.wall_time, tz=UTC)
            try:
                self._redis.xadd(
                    STREAM_KEY,
                    {"payload": json.dumps(payload), "timestamp": timestamp.isoformat()},
                    maxlen=STREAM_MAXLEN,
                    approximate=True,
                )
            except Exception as exc:
                logger.error(f"Redis xadd failed: {exc}")

            self._db_batch.append(
                {
                    "machine_id": reading.machine_id,
                    "timestamp": timestamp,
                    "sensor_name": reading.sensor_name,
                    "value": reading.value,
                    "is_anomaly": False,
                    "machine_phase": reading.phase,
                    "present": reading.present,
                    "fft_data": reading.fft_data,
                }
            )

        if len(self._db_batch) >= DB_FLUSH_BATCH:
            await asyncio.to_thread(self.flush)

    def flush(self) -> None:
        """Bulk-insert buffered readings; drop the batch on hard failure."""
        if not self._db_batch:
            return
        from src.database.models import SensorReading

        batch, self._db_batch = self._db_batch, []
        session = self._session_factory()
        try:
            session.bulk_insert_mappings(SensorReading, batch)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"DB flush of {len(batch)} readings failed: {exc}")
        finally:
            session.close()


async def control_consumer(engine, redis_url: str) -> None:
    """Apply control_stream commands (maintenance, load, injection) to the engine."""
    import redis.asyncio as aioredis

    from src.data_generator.control import CONTROL_STREAM, LINE_MACHINES

    client = aioredis.from_url(redis_url, decode_responses=True)
    last_id = "$"
    while True:
        try:
            messages = await client.xread({CONTROL_STREAM: last_id}, count=10, block=2000)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Control stream read failed: {exc}; retrying in 5s")
            await asyncio.sleep(5)
            continue

        for _stream, events in messages or []:
            for event_id, fields in events:
                last_id = event_id
                command = fields.get("command", "")
                try:
                    if command == "PAUSE_LINE":
                        for mid in LINE_MACHINES.get(fields.get("line", ""), []):
                            engine.pause_machine(mid)
                    elif command == "RESUME_LINE":
                        for mid in LINE_MACHINES.get(fields.get("line", ""), []):
                            engine.resume_machine(mid)
                    elif command == "RESET_MACHINE":
                        engine.reset_machine(fields["machine_id"])
                    elif command == "SET_LOAD":
                        engine.set_load_factor(
                            fields["machine_id"], float(fields.get("factor", 1.0))
                        )
                    elif command == "INJECT_ANOMALY":
                        engine.inject_anomaly(
                            fields["machine_id"],
                            fields.get("scenario", "full_cascade"),
                            ramp_seconds=float(fields.get("ramp_seconds", 10.0)),
                        )
                    # MAINTENANCE_DONE is consumed by the ML service.
                except Exception as exc:
                    logger.error(f"Control command {command} failed: {exc}")


async def main() -> None:
    import redis

    from src.config import settings
    from src.data_generator.independent_scheduler import SimulationEngine
    from src.data_generator.machines import ANOMALY_SCENARIOS, MACHINE_CONFIGS
    from src.database.connection import get_session_factory

    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=False)
    sink = ReadingSink(redis_client, get_session_factory())

    engine = SimulationEngine(
        machine_specs={mid: cfg["sensors"] for mid, cfg in MACHINE_CONFIGS.items()},
        speed_multiplier=float(settings.SIMULATION_SPEED),
        global_seed=settings.GLOBAL_SEED,
        reading_callback=sink,
        anomaly_scenarios=ANOMALY_SCENARIOS,
    )

    logger.info(
        f"Data generator started: {len(MACHINE_CONFIGS)} machines, "
        f"speed={settings.SIMULATION_SPEED}x → {STREAM_KEY} + sensor_readings"
    )
    control_task = asyncio.create_task(control_consumer(engine, settings.REDIS_URL))
    try:
        await engine.run()
    finally:
        control_task.cancel()
        await engine.stop()
        sink.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Data generator stopped")
