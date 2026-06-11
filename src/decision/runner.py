"""Standalone runner for the decision service.

Wires DecisionSubscriber to Redis and the database and drives its
poll/tick loops. Run with: ``python -m src.decision.runner``
"""
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PERIODIC_TICK_SECONDS = 300
OPERATOR_TICK_SECONDS = 5


def run() -> None:
    import redis

    from src.config import settings
    from src.database.connection import get_session_factory
    from src.decision.maintenance import execute_due_jobs, schedule_emergency_repairs
    from src.decision.operator_simulator import act_on_due_decisions
    from src.decision.subscriber import DecisionSubscriber

    demo_mode = os.getenv("DECISION_DEMO_MODE", "true").lower() in ("1", "true", "yes")
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    db = get_session_factory()()

    subscriber = DecisionSubscriber(
        redis_client=redis_client,
        db_session=db,
        demo_mode=demo_mode,
    )
    subscriber._ensure_consumer_group()
    logger.info(
        f"Decision service started (demo_mode={demo_mode}) — consuming anomaly_stream"
    )

    last_periodic = time.monotonic()
    last_operator = time.monotonic()
    try:
        while True:
            subscriber._poll_once()
            now = time.monotonic()
            if now - last_periodic >= PERIODIC_TICK_SECONDS:
                try:
                    subscriber._periodic_tick()
                except Exception as e:
                    logger.error(f"Periodic tick failed: {e}")
                last_periodic = now
            if now - last_operator >= OPERATOR_TICK_SECONDS:
                # Simulated human oversight: after the human response window
                # expires, the on-shift bot of the required rank approves the
                # AI recommendation. Maintenance jobs then run on schedule.
                if demo_mode:
                    try:
                        act_on_due_decisions(db, redis_client)
                    except Exception as e:
                        logger.error(f"Operator simulator tick failed: {e}")
                        db.rollback()
                try:
                    schedule_emergency_repairs(db, redis_client)
                    execute_due_jobs(db, redis_client)
                except Exception as e:
                    logger.error(f"Maintenance tick failed: {e}")
                    db.rollback()
                last_operator = now
    except KeyboardInterrupt:
        logger.info("Decision service stopped")
    finally:
        db.close()


if __name__ == "__main__":
    run()
