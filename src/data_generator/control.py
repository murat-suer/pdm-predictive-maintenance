"""Control channel between the decision layer and the simulation.

The decision service (and the API's demo controls) publish commands on the
Redis ``control_stream``; the data-generator consumes them and drives the
SimulationEngine, while the ML service watches for ``MAINTENANCE_DONE`` to
flush its rolling buffers.

Commands (fields are flat strings):
- ``PAUSE_LINE``      {line}                — production line stops
- ``RESUME_LINE``     {line}                — production line restarts
- ``RESET_MACHINE``   {machine_id}          — maintenance done, full health
- ``SET_LOAD``        {machine_id, factor}  — REDUCE_LOAD throttle
- ``INJECT_ANOMALY``  {machine_id, scenario} — demo fault injection
- ``MAINTENANCE_DONE`` {machine_id}         — ML buffer flush signal
"""
import logging
from datetime import UTC, datetime

from src.data_generator.machines import MACHINE_CONFIGS

logger = logging.getLogger(__name__)

CONTROL_STREAM = "control_stream"

LINE_MACHINES: dict[str, list[str]] = {
    "A": [mid for mid, cfg in MACHINE_CONFIGS.items() if cfg.get("line") == "A"],
    "B": [mid for mid, cfg in MACHINE_CONFIGS.items() if cfg.get("line") == "B"],
}


def machine_line(machine_id: str) -> str:
    return MACHINE_CONFIGS.get(machine_id, {}).get("line", "A")


def publish_control(r, command: str, **fields) -> None:
    """Publish one control command; values are stringified for Redis."""
    payload = {"command": command, "timestamp": datetime.now(UTC).isoformat()}
    payload.update({k: str(v) for k, v in fields.items()})
    try:
        r.xadd(CONTROL_STREAM, payload, maxlen=5000, approximate=True)
        logger.info(f"Control published: {command} {fields}")
    except Exception as exc:
        logger.error(f"Control publish failed ({command}): {exc}")
