"""Shared dependencies for API endpoints."""
from collections.abc import Generator

from sqlalchemy.orm import Session

from src.database.connection import get_db as _get_db

_redis_client = None


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session.

    Yields:
        SQLAlchemy Session with automatic cleanup.
    """
    yield from _get_db()


def get_redis():
    """Dependency that yields a shared Redis client (lazy singleton)."""
    global _redis_client
    if _redis_client is None:
        import redis

        from src.config import settings

        _redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client
