"""Integration test: OBSERVE→DISPATCH_TECHNICIAN escalation through the live path.

Reproduces the production bug where 3 consecutive OBSERVE decisions for
CM-203 resulted in OBSERVE being recommended for the 3rd event instead of
DISPATCH_TECHNICIAN (confirmed from production DB: 63 UNCLASSIFIED anomalies,
zero DISPATCH_TECHNICIAN recommendations over 3 days).

The unit-level apply_observe_escalation tests already pass, so the break is
in the integration between subscriber._process_anomaly_event and the DB query
inside observe_streak / fault_is_identified.
"""
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import (
    AIActLog,
    AlarmState,
    AlarmStateTransition,
    AnomalyLog,
    DecisionAuditLog,
    DecisionLog,
    MachineHealthScore,
    MaintenanceLog,
    WorkOrder,
)

TABLES = (
    AnomalyLog,
    AlarmState,
    AlarmStateTransition,
    DecisionLog,
    DecisionAuditLog,
    WorkOrder,
    MaintenanceLog,
    MachineHealthScore,
    AIActLog,
)

MACHINE_ID = "CM-203"


class GroupFakeRedis:
    """Minimal in-memory Redis stand-in with xgroup_create (needed by subscriber)."""

    def __init__(self):
        self.streams: dict = {}
        self.zsets: dict = {}
        self.kv: dict = {}

    def xgroup_create(self, *a, **k):
        pass

    def xadd(self, stream, fields, **kwargs):
        self.streams.setdefault(stream, []).append(dict(fields))

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def delete(self, key):
        self.kv.pop(key, None)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    def zrangebyscore(self, key, low, high):
        return []

    def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in TABLES:
        table.__table__.create(engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def _seed_approved_observe(db, machine_id: str, fault_type: str, offset_minutes: int):
    """Insert a fully-resolved APPROVE/OBSERVE decision as it would look after
    the bot approves it: AnomalyLog → AlarmState → DecisionLog(APPROVE, OBSERVE).

    ``offset_minutes`` is how many minutes ago the anomaly was created; the
    decision was approved 3 minutes later (matching the ~180 s human window).
    """
    now = datetime.now(UTC)
    detected = now - timedelta(minutes=offset_minutes)
    approved = detected + timedelta(minutes=3)

    anomaly = AnomalyLog(
        machine_id=machine_id,
        detected_at=detected,
        anomaly_score=0.72,
        severity="WARNING",
        status="RESOLVED",  # already closed by close_observation_cycle
        resolution_type="OBSERVE_REVIEW",
        resolved_at=approved,
        fault_type=fault_type,
    )
    db.add(anomaly)
    db.flush()

    alarm = AlarmState(
        anomaly_id=anomaly.id,
        machine_id=machine_id,
        level=1,
        status="NORMAL",  # returned to NORMAL by close_observation_cycle
        created_at=detected,
        last_updated=approved,
    )
    db.add(alarm)
    db.flush()

    # Simulate what the after_insert listener + resolve_decision_plan produce.
    # The DecisionLog has action=APPROVE and chosen_scenario_id=OBSERVE so that
    # observe_streak counts it correctly.
    decision = DecisionLog(
        alarm_id=alarm.id,
        machine_id=machine_id,
        action="APPROVE",
        chosen_scenario_id="OBSERVE",
        decided_at=approved,
        created_at=detected,
        ai_recommendation="OBSERVE",
        auto_approved=True,
    )
    db.add(decision)
    db.commit()
    return anomaly, alarm, decision


class TestObserveEscalationIntegration:
    """The subscriber's live path must escalate to DISPATCH_TECHNICIAN after
    two consecutive OBSERVE decisions for an unclassified anomaly (streak=2,
    identified=False → EEMUA 191 policy)."""

    def test_third_unclassified_anomaly_recommends_dispatch(self, db):
        """Exact reproduction of the CM-203 production sequence:

        1st OBSERVE: UNCLASSIFIED → streak=0 → no escalation (correct)
        2nd OBSERVE: BEARING_FAULT (identified) → streak=1, identified → no forced dispatch (correct)
        3rd anomaly: UNCLASSIFIED → streak=2, not identified → MUST recommend DISPATCH_TECHNICIAN
        """
        # ------------------------------------------------------------------
        # Seed: two prior approved OBSERVE decisions, already committed.
        # Minutes ago chosen to guarantee both are committed before the 3rd
        # anomaly arrives (mirrors production: 23:13:38 and 23:17:24 are both
        # well before 23:19:34).
        # ------------------------------------------------------------------
        _seed_approved_observe(db, MACHINE_ID, "UNCLASSIFIED_ANOMALY", offset_minutes=10)
        _seed_approved_observe(db, MACHINE_ID, "BEARING_FAULT", offset_minutes=6)

        # ------------------------------------------------------------------
        # The 3rd anomaly: inserted by the ML pipeline into anomaly_log BEFORE
        # the event is published to the stream (subscriber reads it by id).
        # ------------------------------------------------------------------
        now = datetime.now(UTC)
        third_anomaly = AnomalyLog(
            machine_id=MACHINE_ID,
            detected_at=now,
            anomaly_score=0.81,
            severity="WARNING",
            status="ACTIVE",
            fault_type="UNCLASSIFIED_ANOMALY",
        )
        db.add(third_anomaly)
        db.commit()

        # Verify the streak is 2 before we invoke the subscriber (sanity check
        # that the seed is correct and observe_streak sees both prior rows).
        from src.decision.observation_policy import fault_is_identified, observe_streak

        assert observe_streak(db, MACHINE_ID) == 2, (
            "Seed incorrect: expected streak=2 from 2 prior OBSERVE/APPROVE rows"
        )
        assert fault_is_identified(db, third_anomaly.id) is False, (
            "Seed incorrect: third anomaly should be unclassified"
        )

        # ------------------------------------------------------------------
        # Drive the REAL integration path: subscriber._process_anomaly_event.
        # ------------------------------------------------------------------
        from src.decision.subscriber import DecisionSubscriber

        sub = DecisionSubscriber(
            redis_client=GroupFakeRedis(),
            db_session=db,
            demo_mode=True,
        )

        # rul_hours=4.0 → DecisionEngine recommends OBSERVE (tested: engine
        # recommends OBSERVE when rul <= 6 h, PLANNED above that).  The
        # production sequence also had an OBSERVE-biased recommendation;
        # the escalation logic must then override it to DISPATCH_TECHNICIAN.
        event = {
            "machine_id": MACHINE_ID,
            "anomaly_id": str(third_anomaly.id),
            "anomaly_score": "0.81",
            "severity": "WARNING",
            "timestamp": now.isoformat(),
            "phase": "DEGRADING",
            "rul_hours": "4.0",
        }
        result = sub._process_anomaly_event(event)

        # The subscriber must return an alarm (not None / suppressed).
        assert result is not None, "subscriber returned None — anomaly processing failed"
        assert not getattr(result, "suppressed", False), "alarm was flood-suppressed unexpectedly"

        # Fetch the resulting DecisionLog via the alarm that was just created.
        decision = (
            db.query(DecisionLog)
            .filter(
                DecisionLog.machine_id == MACHINE_ID,
                DecisionLog.action == "PENDING",
            )
            .order_by(DecisionLog.created_at.desc())
            .first()
        )
        assert decision is not None, "No PENDING DecisionLog was created for the 3rd anomaly"

        # Primary assertion: escalation must fire and recommend DISPATCH.
        assert decision.ai_recommendation == "DISPATCH_TECHNICIAN", (
            f"Expected DISPATCH_TECHNICIAN, got {decision.ai_recommendation!r}. "
            f"The EEMUA 191 escalation policy failed to fire on the 3rd consecutive "
            f"OBSERVE for an unclassified anomaly (streak=2, identified=False)."
        )

        # DISPATCH must also appear in the presented scenario list.
        scenario_ids = [s.get("scenario") for s in (decision.scenarios_presented or [])]
        assert "DISPATCH_TECHNICIAN" in scenario_ids, (
            f"DISPATCH_TECHNICIAN not in scenarios_presented: {scenario_ids}"
        )

        # Exactly one scenario should be flagged as recommended.
        recommended = [s for s in (decision.scenarios_presented or []) if s.get("is_recommended")]
        assert len(recommended) == 1
        assert recommended[0]["scenario"] == "DISPATCH_TECHNICIAN"

    def test_unclassified_streak_overrides_planned_recommendation(self, db):
        """Root cause of the 63-UNCLASSIFIED / ZERO-DISPATCH production gap.

        When rul_hours is high (here: 20 h), the cost model recommends PLANNED.
        The previous implementation only promoted DISPATCH when the model had
        already chosen OBSERVE (``recommendation == "OBSERVE"``).  For
        higher-RUL unclassified anomalies the condition was always False —
        hence ZERO DISPATCH in 3 days.

        The fix: for unidentified faults with a streak, DISPATCH is preferred
        over both OBSERVE and PLANNED (a €150 physical inspection beats
        scheduling expensive maintenance for an unknown fault).
        SHUTDOWN is intentionally exempt — imminent failure takes priority.
        """
        _seed_approved_observe(db, "AC-201", "UNCLASSIFIED_ANOMALY", offset_minutes=12)
        _seed_approved_observe(db, "AC-201", "UNCLASSIFIED_ANOMALY", offset_minutes=8)

        now = datetime.now(UTC)
        anomaly = AnomalyLog(
            machine_id="AC-201",
            detected_at=now,
            anomaly_score=0.65,
            severity="WARNING",
            status="ACTIVE",
            fault_type="UNCLASSIFIED_ANOMALY",
        )
        db.add(anomaly)
        db.commit()

        from src.decision.subscriber import DecisionSubscriber

        sub = DecisionSubscriber(
            redis_client=GroupFakeRedis(),
            db_session=db,
            demo_mode=True,
        )

        # High RUL → engine recommends PLANNED (not OBSERVE).
        # With the old code this would leave the recommendation as PLANNED.
        # With the fix, streak=2 + unidentified → DISPATCH_TECHNICIAN.
        event = {
            "machine_id": "AC-201",
            "anomaly_id": str(anomaly.id),
            "anomaly_score": "0.65",
            "severity": "WARNING",
            "timestamp": now.isoformat(),
            "phase": "DEGRADING",
            "rul_hours": "20.0",
        }
        result = sub._process_anomaly_event(event)

        assert result is not None
        assert not getattr(result, "suppressed", False)

        decision = (
            db.query(DecisionLog)
            .filter(
                DecisionLog.machine_id == "AC-201",
                DecisionLog.action == "PENDING",
            )
            .order_by(DecisionLog.created_at.desc())
            .first()
        )
        assert decision is not None
        assert decision.ai_recommendation == "DISPATCH_TECHNICIAN", (
            f"High-RUL unclassified streak must still yield DISPATCH_TECHNICIAN, "
            f"got {decision.ai_recommendation!r}. This is the production gap that "
            f"produced 63 UNCLASSIFIED anomalies with zero DISPATCH recommendations."
        )
        scenario_ids = [s.get("scenario") for s in (decision.scenarios_presented or [])]
        assert "DISPATCH_TECHNICIAN" in scenario_ids
