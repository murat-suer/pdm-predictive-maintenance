from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = 'postgresql://pdm_user:***@localhost:5432/pdm_intelligence'
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = Field(default=5, ge=1, le=50)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0, le=100)

    # Redis (Phase 3+)
    REDIS_URL: str = 'redis://localhost:6379/0'

    # Simulation. 10x keeps the fleet calm (a machine lives ~3.5 real days)
    # so organic failures are occasional and the demo is driven by fault
    # injection; 500x compresses a full machine life into ~17 minutes.
    SIMULATION_SPEED: int = Field(default=10, ge=1)
    GLOBAL_SEED: int = 42
    SENSOR_PRESENT_PROBABILITY: float = Field(default=0.98, ge=0.0, le=1.0)

    # Maintenance pacing: sim-hour durations are converted to real seconds by
    # SIMULATION_SPEED, then capped so the closed loop stays watchable at low
    # speeds (at 10x an uncapped 4 sim-hour lead would be 24 real minutes).
    MAINT_MAX_REAL_LEAD_S: int = Field(default=120, ge=0)
    MAINT_MAX_REAL_DURATION_S: int = Field(default=120, ge=1)

    # API: comma-separated allowed CORS origins; "*" for local development,
    # the public dashboard origin in production.
    CORS_ALLOW_ORIGINS: str = '*'

    # Logging
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'

    # Retention
    DATA_RETENTION_DAYS: int = Field(default=90, ge=1)
    COMPRESSION_AFTER_DAYS: int = Field(default=7, ge=1)

    # Decision: human response window before the simulated operator acts
    # (3 min nominal, ±20% jitter applied at decision creation).
    AUTO_APPROVE_DELAY_SECONDS: int = Field(default=180, ge=0)

    model_config = {'env_file': '.env', 'env_file_encoding': 'utf-8', 'extra': 'ignore'}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
