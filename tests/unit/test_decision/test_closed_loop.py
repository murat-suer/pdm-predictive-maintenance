"""Closed-loop tests: operator simulator + maintenance lifecycle."""
import json
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
from src.decision.maintenance import (
    SCHEDULE_KEY,
    execute_due_jobs,
    schedule_from_decision,
)
from src.decision.operator_simulator import (
    act_on_due_decisions,
    bot_identity,
    current_shift,
    required_rank,
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


class FakeRedis:
    """Minimal in-memory stand-in for streams + sorted sets."""

    def __init__(self):
        self.streams: dict[str, list[dict]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.kv: dict[str, str] = {}

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
        entries = self.zsets.get(key, {})
        high_val = float("inf") if high in ("+inf", "inf") else float(high)
        low_val = float("-inf") if low in ("-inf",) else float(low)
        return [m for m, score in entries.items() if low_val <= score <= high_val]

    def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)

    def commands(self, stream="control_stream"):
        return [f.get("command") for f in self.streams.get(stream, [])]


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


@pytest.fixture
def redis_client():
    return FakeRedis()


def seed_pending_decision(db, machine_id="AC-201", recommendation="PLANNED", due_offset_s=-10):
    now = datetime.now(UTC)
    anomaly = AnomalyLog(
        machine_id=machine_id,
        detected_at=now,
        anomaly_score=0.9,
        severity="CRITICAL",
        status="ACTIVE",
        fault_type="BEARING_FAULT",
    )
    db.add(anomaly)
    db.flush()
    alarm = AlarmState(
        anomaly_id=anomaly.id,
        machine_id=machine_id,
        level=2,
        status="UNACKNOWLEDGED",
        created_at=now,
        last_updated=now,
    )
    db.add(alarm)
    db.flush()
    decision = db.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
    decision.ai_recommendation = recommendation
    decision.scenarios_presented = [
        {"scenario": "OBSERVE", "cost": 0.0, "expected_cost": 90000.0, "is_recommended": False},
        {"scenario": recommendation, "cost": 1200.0, "expected_cost": 4000.0, "is_recommended": True},
    ]
    decision.notes = json.dumps({"run_to_failure_cost": 25000.0, "rul_hours": 12.0})
    decision.due_at = now + timedelta(seconds=due_offset_s)
    db.commit()
    return decision, alarm


class TestShiftsAndRanks:
    def test_shift_boundaries(self):
        assert current_shift(datetime(2026, 6, 10, 7, 0, tzinfo=UTC)) == "ALPHA"
        assert current_shift(datetime(2026, 6, 10, 15, 0, tzinfo=UTC)) == "BRAVO"
        assert current_shift(datetime(2026, 6, 10, 23, 0, tzinfo=UTC)) == "CHARLIE"
        assert current_shift(datetime(2026, 6, 10, 2, 0, tzinfo=UTC)) == "CHARLIE"

    def test_rank_thresholds(self):
        assert required_rank(100) == "OPERATOR"
        assert required_rank(1500) == "SUPERVISOR"
        assert required_rank(50000) == "MANAGER"

    def test_rank_exact_boundaries(self):
        """Authority limits are exclusive upper bounds."""
        assert required_rank(499.99) == "OPERATOR"
        assert required_rank(500.0) == "SUPERVISOR"
        assert required_rank(1999.99) == "SUPERVISOR"
        assert required_rank(2000.0) == "MANAGER"

    def test_bot_identity_encodes_rank_and_shift(self):
        operator_id, role = bot_identity(1500, datetime(2026, 6, 10, 15, 0, tzinfo=UTC))
        assert operator_id == "BOT-SUP-BRAVO"
        assert role == "SUPERVISOR"


class TestOperatorSimulator:
    def test_bot_waits_for_human_window(self, db, redis_client):
        seed_pending_decision(db, due_offset_s=+300)  # window still open
        resolved = act_on_due_decisions(db, redis_client)
        assert resolved == []

    def test_bot_approves_recommendation_after_window(self, db, redis_client):
        decision, alarm = seed_pending_decision(db, due_offset_s=-10)
        resolved = act_on_due_decisions(db, redis_client)
        assert len(resolved) == 1
        db.refresh(decision)
        assert decision.action == "APPROVE"
        assert decision.chosen_scenario_id == "PLANNED"
        assert decision.auto_approved is True
        assert decision.decided_by.startswith("BOT-")
        # 1200 EUR action → SUPERVISOR authority
        assert decision.decided_by.startswith("BOT-SUP-")
        assert decision.operator_role == "SUPERVISOR"
        # full chain: audit row written, alarm shelved for PLANNED
        assert db.query(DecisionAuditLog).count() == 1
        db.refresh(alarm)
        assert alarm.status == "SHELVED"

    def test_bot_schedules_maintenance(self, db, redis_client):
        seed_pending_decision(db, due_offset_s=-10)
        act_on_due_decisions(db, redis_client)
        assert len(redis_client.zsets.get(SCHEDULE_KEY, {})) == 1
        assert db.query(WorkOrder).count() == 1


class TestMaintenanceLifecycle:
    def _approved_decision(self, db, redis_client, scenario="PLANNED"):
        decision, alarm = seed_pending_decision(db, recommendation=scenario)
        act_on_due_decisions(db, redis_client)
        db.refresh(decision)
        db.refresh(alarm)
        return decision, alarm

    def test_shutdown_starts_immediately(self, db, redis_client):
        self._approved_decision(db, redis_client, scenario="SHUTDOWN")
        executed = execute_due_jobs(db, redis_client)
        assert executed == 1
        assert "PAUSE_LINE" in redis_client.commands()

    def test_start_pauses_line_and_marks_oos(self, db, redis_client):
        decision, alarm = self._approved_decision(db, redis_client, scenario="SHUTDOWN")
        execute_due_jobs(db, redis_client)
        db.refresh(alarm)
        assert alarm.status == "OUT_OF_SERVICE"
        work_order = db.query(WorkOrder).first()
        assert work_order.status == "IN_PROGRESS"
        # completion entry is queued for later
        assert len(redis_client.zsets[SCHEDULE_KEY]) == 1

    def test_complete_restores_machine_and_accounts(self, db, redis_client):
        decision, alarm = self._approved_decision(db, redis_client, scenario="SHUTDOWN")
        execute_due_jobs(db, redis_client)  # START
        # Force the completion entry due now
        key = next(iter(redis_client.zsets[SCHEDULE_KEY]))
        redis_client.zsets[SCHEDULE_KEY][key] = 0.0
        execute_due_jobs(db, redis_client)  # COMPLETE

        commands = redis_client.commands()
        assert "RESET_MACHINE" in commands
        assert "RESUME_LINE" in commands
        assert "MAINTENANCE_DONE" in commands

        db.refresh(alarm)
        assert alarm.status == "NORMAL"
        assert alarm.oos_restored_at is not None

        anomaly = db.query(AnomalyLog).first()
        assert anomaly.status == "RESOLVED"
        assert anomaly.resolution_type == "SHUTDOWN"

        work_order = db.query(WorkOrder).first()
        assert work_order.status == "COMPLETED"

        log = db.query(MaintenanceLog).first()
        assert log is not None
        assert log.cost_eur == 1200.0

        db.refresh(decision)
        assert decision.outcome == "MAINTENANCE_COMPLETED"

    def test_observe_schedules_nothing(self, db, redis_client):
        decision, alarm = seed_pending_decision(db, recommendation="OBSERVE")
        act_on_due_decisions(db, redis_client)
        schedule_from_decision(db, redis_client, decision, alarm)
        assert redis_client.zsets.get(SCHEDULE_KEY, {}) == {}

    def test_reduce_load_throttles_machine(self, db, redis_client):
        seed_pending_decision(db, recommendation="REDUCE_LOAD")
        act_on_due_decisions(db, redis_client)
        set_load = [
            f for f in redis_client.streams.get("control_stream", [])
            if f.get("command") == "SET_LOAD"
        ]
        assert len(set_load) == 1
        assert set_load[0]["factor"] == "0.8"


class TestEmergencyRepair:
    def test_failed_machine_gets_emergency_repair(self, db, redis_client):
        from src.database.models import SensorReading
        from src.decision.maintenance import schedule_emergency_repairs

        SensorReading.__table__.create(db.get_bind(), checkfirst=True)
        db.add(
            SensorReading(
                machine_id="CM-203",
                timestamp=datetime.now(UTC),
                sensor_name="belt_tension",
                value=0.1,
                machine_phase="FAILED",
            )
        )
        db.commit()

        scheduled = schedule_emergency_repairs(db, redis_client)
        assert scheduled == 1
        jobs = [json.loads(j) for j in redis_client.zsets[SCHEDULE_KEY]]
        assert jobs[0]["scenario"] == "EMERGENCY"
        assert jobs[0]["machine_id"] == "CM-203"
        # idempotent: lock prevents double dispatch
        assert schedule_emergency_repairs(db, redis_client) == 0

    def test_emergency_completion_books_reactive_cost(self, db, redis_client):
        from src.database.models import SensorReading
        from src.decision.maintenance import schedule_emergency_repairs

        SensorReading.__table__.create(db.get_bind(), checkfirst=True)
        db.add(
            SensorReading(
                machine_id="CM-203",
                timestamp=datetime.now(UTC),
                sensor_name="belt_tension",
                value=0.1,
                machine_phase="FAILED",
            )
        )
        db.commit()
        schedule_emergency_repairs(db, redis_client)
        execute_due_jobs(db, redis_client)  # START
        key = next(iter(redis_client.zsets[SCHEDULE_KEY]))
        redis_client.zsets[SCHEDULE_KEY][key] = 0.0
        execute_due_jobs(db, redis_client)  # COMPLETE

        log = db.query(MaintenanceLog).first()
        assert log is not None
        # full reactive cost, far above a planned repair for a conveyor
        assert log.cost_eur > 5000
        assert "run-to-failure" in log.technician_notes


class TestObservationCycle:
    def test_observe_closes_anomaly_for_reevaluation(self, db, redis_client):
        decision, alarm = seed_pending_decision(db, recommendation="OBSERVE")
        act_on_due_decisions(db, redis_client)
        db.refresh(alarm)
        # alarm returns to NORMAL and the anomaly is closed, so the ML
        # pipeline can raise a fresh alarm with an updated RUL later
        assert alarm.status == "NORMAL"
        anomaly = db.query(AnomalyLog).first()
        assert anomaly.status == "RESOLVED"
        assert anomaly.resolution_type == "OBSERVE_REVIEW"


class TestMachineWideAlarmClosure:
    def test_completion_closes_orphaned_alarms(self, db, redis_client):
        """An alarm raised by a parallel detection wave (e.g. SHELVED while
        an emergency repair was already in flight) must not survive the
        machine's full-health restoration."""
        from datetime import UTC, datetime

        decision, alarm = seed_pending_decision(db, machine_id="AC-201", recommendation="SHUTDOWN")
        act_on_due_decisions(db, redis_client)
        # Orphan: a second alarm with no job linkage, shelved mid-repair.
        orphan_anomaly = AnomalyLog(
            machine_id="AC-201",
            detected_at=datetime.now(UTC),
            anomaly_score=0.8,
            severity="WARNING",
            status="ACTIVE",
            fault_type="OIL_DEGRADATION",
        )
        db.add(orphan_anomaly)
        db.flush()
        orphan = AlarmState(
            anomaly_id=orphan_anomaly.id,
            machine_id="AC-201",
            level=2,
            status="SHELVED",
            created_at=datetime.now(UTC),
            last_updated=datetime.now(UTC),
        )
        db.add(orphan)
        db.commit()

        execute_due_jobs(db, redis_client)  # START
        key = next(iter(redis_client.zsets[SCHEDULE_KEY]))
        redis_client.zsets[SCHEDULE_KEY][key] = 0.0
        execute_due_jobs(db, redis_client)  # COMPLETE

        db.refresh(orphan)
        assert orphan.status == "NORMAL"
        assert orphan.oos_restored_at is not None


class TestOrphanReconciliation:
    def test_orphan_anomaly_gets_alarm_and_decision(self, db, redis_client):
        """A lost stream event must not silence the machine: the sweep
        re-creates the alarm and decision from the DB row."""
        from datetime import UTC, datetime, timedelta

        from src.decision.subscriber import DecisionSubscriber

        orphan = AnomalyLog(
            machine_id="HX-202",
            detected_at=datetime.now(UTC) - timedelta(minutes=5),
            anomaly_score=0.52,
            severity="WARNING",
            status="ACTIVE",
            fault_type="FOULING",
        )
        db.add(orphan)
        db.commit()

        class GroupFakeRedis(FakeRedis):
            def xgroup_create(self, *a, **k):
                pass

        sub = DecisionSubscriber(
            redis_client=GroupFakeRedis(), db_session=db, demo_mode=True
        )
        healed = sub.reconcile_orphan_anomalies()
        assert healed == 1

        alarm = db.query(AlarmState).filter(AlarmState.machine_id == "HX-202").first()
        assert alarm is not None
        assert alarm.anomaly_id == orphan.id
        decision = db.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        assert decision is not None
        assert decision.ai_recommendation is not None

    def test_recent_anomalies_left_alone(self, db, redis_client):
        """An anomaly younger than the grace window is the stream's job."""
        from datetime import UTC, datetime

        from src.decision.subscriber import DecisionSubscriber

        fresh = AnomalyLog(
            machine_id="HX-202",
            detected_at=datetime.now(UTC),
            anomaly_score=0.52,
            severity="WARNING",
            status="ACTIVE",
            fault_type="FOULING",
        )
        db.add(fresh)
        db.commit()

        class GroupFakeRedis(FakeRedis):
            def xgroup_create(self, *a, **k):
                pass

        sub = DecisionSubscriber(
            redis_client=GroupFakeRedis(), db_session=db, demo_mode=True
        )
        assert sub.reconcile_orphan_anomalies() == 0
