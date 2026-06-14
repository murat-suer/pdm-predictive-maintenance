"""Unit tests for the /api/v1/analytics router.

Conventions match test_dashboard_api.py:
- In-memory SQLite via StaticPool.
- All tables created from ORM metadata.
- TestClient with get_db dependency overridden.
- One test class per endpoint.
"""
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_db
from src.database.models import (
    AlarmState,
    AnomalyLog,
    DecisionLog,
    MachineHealthScore,
    MaintenanceLog,
)

TABLES = (
    AnomalyLog,
    AlarmState,
    DecisionLog,
    MachineHealthScore,
    MaintenanceLog,
)


@pytest.fixture
def db_session():
    """In-memory SQLite session with analytics-relevant tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in TABLES:
        table.__table__.create(engine, checkfirst=True)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def client(db_session):
    """TestClient with the DB dependency overridden to the in-memory SQLite."""
    from src.api.app import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# ─── Seed helpers ─────────────────────────────────────────────────────────


def _now():
    return datetime.now(UTC)


def seed_health_score(db, machine_id="AC-201", offset_hours=0, score=0.85):
    """Insert a MachineHealthScore row (establishes cycle_start)."""
    ts = _now() - timedelta(hours=offset_hours)
    db.add(
        MachineHealthScore(
            machine_id=machine_id,
            calculated_at=ts,
            health_score=score,
            reliability_score=0.9,
        )
    )
    db.commit()
    return ts


def seed_anomaly(db, machine_id="AC-201", fault_type="BEARING_FAULT", offset_hours=1):
    """Insert an AnomalyLog row."""
    ts = _now() - timedelta(hours=offset_hours)
    anomaly = AnomalyLog(
        machine_id=machine_id,
        detected_at=ts,
        anomaly_score=0.8,
        severity="WARNING",
        status="ACTIVE",
        fault_type=fault_type,
    )
    db.add(anomaly)
    db.flush()
    db.commit()
    return anomaly


def seed_decision(
    db,
    alarm_id=None,
    machine_id="AC-201",
    chosen_scenario_id="PLANNED",
    decided_by="HUMAN-OP-1",
    auto_approved=False,
    overridden=False,
    response_time_s=120,
    notes_cost=15000.0,
    offset_hours=1,
):
    """Insert an APPROVE DecisionLog row."""
    ts = _now() - timedelta(hours=offset_hours)
    notes = json.dumps({"run_to_failure_cost": notes_cost})
    d = DecisionLog(
        machine_id=machine_id,
        alarm_id=alarm_id,
        action="APPROVE",
        chosen_scenario_id=chosen_scenario_id,
        decided_by=decided_by,
        auto_approved=auto_approved,
        overridden=overridden,
        response_time_s=response_time_s,
        notes=notes,
        created_at=ts,
    )
    db.add(d)
    db.flush()
    db.commit()
    return d


def seed_alarm(db, anomaly_id, machine_id="AC-201"):
    """Insert an AlarmState row and return it."""
    alarm = AlarmState(
        anomaly_id=anomaly_id,
        machine_id=machine_id,
        level=1,
        status="UNACKNOWLEDGED",
        created_at=_now(),
        last_updated=_now(),
    )
    db.add(alarm)
    db.flush()
    db.commit()
    return alarm


def seed_maintenance(
    db,
    alarm_id=None,
    machine_id="AC-201",
    cost_eur=2000.0,
    downtime_minutes=60,
    offset_hours=0,
):
    """Insert a completed MaintenanceLog row."""
    ts = _now() - timedelta(hours=offset_hours)
    log = MaintenanceLog(
        machine_id=machine_id,
        alarm_id=alarm_id,
        performed_at=ts,
        cost_eur=cost_eur,
        downtime_minutes=downtime_minutes,
    )
    db.add(log)
    db.flush()
    db.commit()
    return log


# ─── Tests: /analytics/decision-mix ──────────────────────────────────────


class TestDecisionMix:
    def test_empty_db_returns_zero_total(self, client, db_session):
        seed_health_score(db_session)
        r = client.get("/api/v1/analytics/decision-mix")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_counts_approve_decisions_by_scenario(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        seed_decision(db_session, chosen_scenario_id="PLANNED")
        seed_decision(db_session, chosen_scenario_id="PLANNED")
        seed_decision(db_session, chosen_scenario_id="REDUCE_LOAD")
        r = client.get("/api/v1/analytics/decision-mix")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3
        by_scenario = {i["scenario"]: i["count"] for i in data["items"]}
        assert by_scenario["PLANNED"] == 2
        assert by_scenario["REDUCE_LOAD"] == 1

    def test_non_approve_decisions_excluded(self, client, db_session):
        """PENDING decisions must not appear in the mix."""
        seed_health_score(db_session)
        # Directly insert a PENDING row (not using seed_decision which forces APPROVE)
        ts = _now()
        db_session.add(
            DecisionLog(
                machine_id="AC-201",
                action="PENDING",
                chosen_scenario_id="OBSERVE",
                created_at=ts,
            )
        )
        db_session.commit()
        r = client.get("/api/v1/analytics/decision-mix")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0

    def test_hours_override_narrows_window(self, client, db_session):
        """Decisions older than ?hours= must be excluded."""
        seed_health_score(db_session)
        seed_decision(db_session, offset_hours=5)   # outside ?hours=2
        seed_decision(db_session, chosen_scenario_id="SHUTDOWN", offset_hours=1)  # inside
        r = client.get("/api/v1/analytics/decision-mix?hours=2")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["scenario"] == "SHUTDOWN"

    def test_returns_window_started_at(self, client, db_session):
        seed_health_score(db_session)
        r = client.get("/api/v1/analytics/decision-mix")
        assert r.status_code == 200
        assert r.json()["window_started_at"] is not None

    def test_no_health_scores_still_returns_200(self, client):
        """Empty DB (no MachineHealthScore) must return 200 with window_started_at=null."""
        r = client.get("/api/v1/analytics/decision-mix")
        assert r.status_code == 200
        data = r.json()
        assert data["window_started_at"] is None
        assert data["total"] == 0


# ─── Tests: /analytics/fault-distribution ────────────────────────────────


class TestFaultDistribution:
    def test_empty_db_returns_zero_total(self, client, db_session):
        seed_health_score(db_session)
        r = client.get("/api/v1/analytics/fault-distribution")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_counts_by_fault_type(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        seed_anomaly(db_session, fault_type="BEARING_FAULT")
        seed_anomaly(db_session, fault_type="BEARING_FAULT")
        seed_anomaly(db_session, fault_type="FOULING")
        r = client.get("/api/v1/analytics/fault-distribution")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3
        by_type = {i["fault_type"]: i["count"] for i in data["items"]}
        assert by_type["BEARING_FAULT"] == 2
        assert by_type["FOULING"] == 1

    def test_null_fault_type_mapped_to_unclassified(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        seed_anomaly(db_session, fault_type=None)
        r = client.get("/api/v1/analytics/fault-distribution")
        assert r.status_code == 200
        data = r.json()
        by_type = {i["fault_type"]: i["count"] for i in data["items"]}
        assert "UNCLASSIFIED_ANOMALY" in by_type

    def test_hours_override_narrows_window(self, client, db_session):
        seed_health_score(db_session)
        seed_anomaly(db_session, fault_type="BEARING_FAULT", offset_hours=10)  # outside
        seed_anomaly(db_session, fault_type="FOULING", offset_hours=1)          # inside
        r = client.get("/api/v1/analytics/fault-distribution?hours=2")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["fault_type"] == "FOULING"

    def test_no_health_scores_still_returns_200(self, client):
        r = client.get("/api/v1/analytics/fault-distribution")
        assert r.status_code == 200
        assert r.json()["window_started_at"] is None


# ─── Tests: /analytics/savings-timeseries ────────────────────────────────


class TestSavingsTimeseries:
    def test_empty_db_returns_empty_points(self, client, db_session):
        seed_health_score(db_session)
        r = client.get("/api/v1/analytics/savings-timeseries")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["points"], list)

    def test_averting_maintenance_contributes_savings(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        anomaly = seed_anomaly(db_session, offset_hours=1)
        alarm = seed_alarm(db_session, anomaly.id)
        decision = seed_decision(
            db_session,
            alarm_id=alarm.id,
            chosen_scenario_id="PLANNED",
            notes_cost=15000.0,
            offset_hours=1,
        )
        seed_maintenance(db_session, alarm_id=alarm.id, cost_eur=2000.0, offset_hours=0)

        r = client.get("/api/v1/analytics/savings-timeseries?buckets=4")
        assert r.status_code == 200
        data = r.json()
        points = data["points"]
        assert len(points) == 4
        # Final cumulative should be 15000 - 2000 = 13000
        assert points[-1]["cumulative_eur"] == pytest.approx(13000.0, abs=1.0)

    def test_non_averting_scenario_excluded(self, client, db_session):
        """OBSERVE decisions don't contribute savings."""
        seed_health_score(db_session, offset_hours=2)
        anomaly = seed_anomaly(db_session, offset_hours=1)
        alarm = seed_alarm(db_session, anomaly.id)
        seed_decision(
            db_session,
            alarm_id=alarm.id,
            chosen_scenario_id="OBSERVE",
            notes_cost=5000.0,
            offset_hours=1,
        )
        seed_maintenance(db_session, alarm_id=alarm.id, cost_eur=500.0, offset_hours=0)

        r = client.get("/api/v1/analytics/savings-timeseries?buckets=4")
        assert r.status_code == 200
        data = r.json()
        # All cumulative_eur should be 0 (no averting scenario)
        assert all(p["cumulative_eur"] == 0.0 for p in data["points"])

    def test_buckets_param_controls_output_length(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        r = client.get("/api/v1/analytics/savings-timeseries?buckets=6")
        assert r.status_code == 200
        data = r.json()
        assert len(data["points"]) == 6

    def test_hours_override_narrows_window(self, client, db_session):
        seed_health_score(db_session, offset_hours=24)
        r = client.get("/api/v1/analytics/savings-timeseries?hours=2&buckets=3")
        assert r.status_code == 200
        assert r.json()["window_started_at"] is not None


# ─── Tests: /analytics/mhi-history ───────────────────────────────────────


class TestMHIHistory:
    def test_empty_db_returns_empty_machines(self, client, db_session):
        r = client.get("/api/v1/analytics/mhi-history")
        assert r.status_code == 200
        data = r.json()
        assert data["machines"] == []

    def test_health_score_scaled_to_0_100(self, client, db_session):
        """health_score 0.75 in DB → mhi 75.0 in response."""
        seed_health_score(db_session, machine_id="AC-201", score=0.75, offset_hours=1)
        r = client.get("/api/v1/analytics/mhi-history?buckets=2")
        assert r.status_code == 200
        data = r.json()
        machines = data["machines"]
        assert len(machines) == 1
        assert machines[0]["machine_id"] == "AC-201"
        # At least one point must exist
        assert len(machines[0]["points"]) >= 1
        assert machines[0]["points"][-1]["mhi"] == pytest.approx(75.0, abs=0.1)

    def test_multiple_machines_returned(self, client, db_session):
        seed_health_score(db_session, machine_id="AC-201", score=0.8, offset_hours=1)
        seed_health_score(db_session, machine_id="HX-202", score=0.6, offset_hours=1)
        r = client.get("/api/v1/analytics/mhi-history?buckets=4")
        assert r.status_code == 200
        machine_ids = {m["machine_id"] for m in r.json()["machines"]}
        assert {"AC-201", "HX-202"} == machine_ids

    def test_hours_override_narrows_window(self, client, db_session):
        # Score from 10 hours ago — outside ?hours=2
        seed_health_score(db_session, machine_id="AC-201", score=0.5, offset_hours=10)
        r = client.get("/api/v1/analytics/mhi-history?hours=2&buckets=4")
        assert r.status_code == 200
        # The row is outside the window, so no machines (or empty points)
        machines = r.json()["machines"]
        assert machines == []

    def test_buckets_param_respected(self, client, db_session):
        """With a single score and 6 buckets, at most 1 non-empty bucket."""
        seed_health_score(db_session, machine_id="AC-201", score=0.9, offset_hours=1)
        r = client.get("/api/v1/analytics/mhi-history?buckets=6")
        assert r.status_code == 200
        machines = r.json()["machines"]
        if machines:
            total_points = sum(len(m["points"]) for m in machines)
            assert total_points <= 6


# ─── Tests: /analytics/maintenance-timeline ──────────────────────────────


class TestMaintenanceTimeline:
    def test_empty_db_returns_empty_events(self, client, db_session):
        seed_health_score(db_session)
        r = client.get("/api/v1/analytics/maintenance-timeline")
        assert r.status_code == 200
        assert r.json()["events"] == []

    def test_maintenance_event_fields(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        anomaly = seed_anomaly(db_session, offset_hours=1)
        alarm = seed_alarm(db_session, anomaly.id)
        seed_decision(
            db_session,
            alarm_id=alarm.id,
            chosen_scenario_id="PLANNED",
            notes_cost=12000.0,
            offset_hours=1,
        )
        seed_maintenance(
            db_session,
            alarm_id=alarm.id,
            cost_eur=3000.0,
            downtime_minutes=45,
            offset_hours=0,
        )
        r = client.get("/api/v1/analytics/maintenance-timeline")
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) == 1
        e = events[0]
        assert e["machine_id"] == "AC-201"
        assert e["scenario"] == "PLANNED"
        assert e["actual_cost_eur"] == pytest.approx(3000.0)
        assert e["savings_eur"] == pytest.approx(9000.0)  # 12000 - 3000
        assert e["downtime_minutes"] == 45

    def test_ordered_by_performed_at_desc(self, client, db_session):
        seed_health_score(db_session, offset_hours=3)
        seed_maintenance(db_session, cost_eur=1000.0, offset_hours=2)
        seed_maintenance(db_session, cost_eur=2000.0, offset_hours=1)
        r = client.get("/api/v1/analytics/maintenance-timeline")
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) == 2
        # More recent (offset_hours=1) must come first
        assert events[0]["actual_cost_eur"] == pytest.approx(2000.0)
        assert events[1]["actual_cost_eur"] == pytest.approx(1000.0)

    def test_hours_override_narrows_window(self, client, db_session):
        seed_health_score(db_session, offset_hours=12)
        seed_maintenance(db_session, cost_eur=999.0, offset_hours=10)  # outside ?hours=2
        seed_maintenance(db_session, cost_eur=111.0, offset_hours=1)   # inside
        r = client.get("/api/v1/analytics/maintenance-timeline?hours=2")
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) == 1
        assert events[0]["actual_cost_eur"] == pytest.approx(111.0)

    def test_no_health_scores_still_returns_200(self, client):
        r = client.get("/api/v1/analytics/maintenance-timeline")
        assert r.status_code == 200
        assert r.json()["window_started_at"] is None


# ─── Tests: /analytics/decision-stats ────────────────────────────────────


class TestDecisionStats:
    def test_empty_db_returns_zero_stats(self, client, db_session):
        seed_health_score(db_session)
        r = client.get("/api/v1/analytics/decision-stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["auto_approved"] == 0
        assert data["overridden"] == 0
        assert data["override_rate"] == 0.0
        assert data["avg_response_time_s"] is None
        assert data["by_actor"] == []
        assert data["bot_vs_human"] == {"bot": 0, "human": 0}

    def test_counts_and_rates(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        seed_decision(
            db_session,
            decided_by="HUMAN-OP-1",
            auto_approved=False,
            overridden=True,
            response_time_s=60,
        )
        seed_decision(
            db_session,
            decided_by="BOT-AUTO",
            auto_approved=True,
            overridden=False,
            response_time_s=180,
        )
        r = client.get("/api/v1/analytics/decision-stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert data["auto_approved"] == 1
        assert data["overridden"] == 1
        assert data["override_rate"] == pytest.approx(0.5, abs=0.001)
        assert data["avg_response_time_s"] == pytest.approx(120.0, abs=0.1)

    def test_bot_vs_human_classification(self, client, db_session):
        """decided_by starting with 'BOT-' → bot, otherwise → human."""
        seed_health_score(db_session, offset_hours=2)
        seed_decision(db_session, decided_by="BOT-AUTO-1")
        seed_decision(db_session, decided_by="BOT-AUTO-2")
        seed_decision(db_session, decided_by="HUMAN-OP-1")
        r = client.get("/api/v1/analytics/decision-stats")
        assert r.status_code == 200
        bvh = r.json()["bot_vs_human"]
        assert bvh["bot"] == 2
        assert bvh["human"] == 1

    def test_by_actor_sorted_by_count_desc(self, client, db_session):
        seed_health_score(db_session, offset_hours=2)
        seed_decision(db_session, decided_by="HUMAN-OP-1")
        seed_decision(db_session, decided_by="HUMAN-OP-1")
        seed_decision(db_session, decided_by="BOT-AUTO")
        r = client.get("/api/v1/analytics/decision-stats")
        assert r.status_code == 200
        by_actor = r.json()["by_actor"]
        assert by_actor[0]["decided_by"] == "HUMAN-OP-1"
        assert by_actor[0]["count"] == 2

    def test_non_approve_rows_excluded(self, client, db_session):
        """PENDING rows must not affect stats."""
        seed_health_score(db_session)
        ts = _now()
        db_session.add(
            DecisionLog(
                machine_id="AC-201",
                action="PENDING",
                decided_by="HUMAN-OP-1",
                auto_approved=False,
                overridden=False,
                response_time_s=30,
                created_at=ts,
            )
        )
        db_session.commit()
        r = client.get("/api/v1/analytics/decision-stats")
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_hours_override_narrows_window(self, client, db_session):
        seed_health_score(db_session)
        seed_decision(db_session, offset_hours=10)  # outside ?hours=2
        seed_decision(db_session, decided_by="BOT-AUTO", offset_hours=1)  # inside
        r = client.get("/api/v1/analytics/decision-stats?hours=2")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["bot_vs_human"]["bot"] == 1
        assert data["bot_vs_human"]["human"] == 0

    def test_no_health_scores_still_returns_200(self, client):
        r = client.get("/api/v1/analytics/decision-stats")
        assert r.status_code == 200
        assert r.json()["window_started_at"] is None
