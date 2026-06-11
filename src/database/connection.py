"""Database connection management with lazy initialization and retry logic."""
from collections.abc import Generator
from contextlib import contextmanager

import sqlalchemy.exc
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings

retry_on_operational_error = retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(sqlalchemy.exc.OperationalError),
    reraise=True,
)


def with_retry(func):
    """Wrap a callable with retry logic for transient database errors.

    Usage:
        @with_retry
        def my_db_operation():
            ...

    Or:
        retried_func = with_retry(some_func)
        result = retried_func()
    """
    return retry_on_operational_error(func)


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    """Get or create the database engine (lazy initialization)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.DB_ECHO,
            connect_args={"connect_timeout": 10},
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


@retry_on_operational_error
def get_db() -> Generator[Session, None, None]:
    """Get database session with retry logic for transient failures."""
    db = get_session_factory()()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database session with automatic commit/rollback."""
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def dispose_engine() -> None:
    """Dispose the engine and reset state for re-creation."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
