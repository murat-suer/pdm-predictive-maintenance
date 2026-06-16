"""Unit tests for the database-backed dashboard API endpoints."""
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_db
from src.database.models import (
    AlarmState,
    AlarmStateTransition,
    AnomalyLog,
    DecisionAuditLog,
    DecisionLog,
    MachineHealthScore,
    MaintenanceLog,
    SensorReading,
    ShiftReport,
    WorkOrder,
)

TABLES = (
    AnomalyLog,
    AlarmState,
    AlarmStateTransition,
    DecisionLog,
    DecisionAuditLog,
    MachineHealthScore,
    MaintenanceLog,
    SensorReading,
    WorkOrder,
    ShiftReport,
)


@pytest.fixture
def db_session():
    """In-memory SQLite session with all dashboard tables."""
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
    """TestClient with the DB dependency overridden to SQLite."""
    from src.api.app import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def seed_machine_data(db, machine_id="AC-201", ai_recommendation: str | None = None):
    """Insert one healthy reading, health score, anomaly and alarm.

    Optionally sets ``ai_recommendation`` on the auto-created DecisionLog so
    status tests can assert recommendation-based tiers.
    """
    now = datetime.now(UTC)
    db.add(
        SensorReading(
            machine_id=machine_id,
            timestamp=now,
            sensor_name="vibration_rms",
            value=4.9,
            is_anomaly=True,
            anomaly_score=0.82,
            machine_phase="DEGRADING",
        )
    )
    db.add(
        MachineHealthScore(
            machine_id=machine_id,
            calculated_at=now,
            health_score=0.71,
            availability_score=0.97,
            reliability_score=0.84,
            condition_score=0.65,
            rul_hours=96.0,
            confidence=0.9,
            classification="Degrading",
        )
    )
    anomaly = AnomalyLog(
        machine_id=machine_id,
        detected_at=now,
        anomaly_score=0.82,
        severity="CRITICAL",
        status="ACTIVE",
        fault_type="BEARING_FAULT",
        top_contributing_sensor="vibration_rms",
        shap_values={"vibration_rms_value": 0.42},
    )
    db.add(anomaly)
    db.flush()
    alarm = AlarmState(
        anomaly_id=anomaly.id,
        machine_id=machine_id,
        level=2,
        status="UNACKNOWLEDGED",
        created_at=now - timedelta(minutes=30),
        last_updated=now,
    )
    db.add(alarm)
    db.flush()
    # The after_insert listener auto-creates a PENDING DecisionLog; set the
    # recommendation if requested (bypasses the listener's default None).
    if ai_recommendation is not None:
        decision = db.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        if decision is not None:
            decision.ai_recommendation = ai_recommendation
        else:
            # Listener did not fire (e.g. in certain SQLite setups) — create manually.
            db.add(
                DecisionLog(
                    alarm_id=alarm.id,
                    machine_id=machine_id,
                    action="PENDING",
                    ai_recommendation=ai_recommendation,
                    created_at=now,
                )
            )
    db.commit()
    return alarm


class TestMachinesEndpoints:
    def test_list_machines_returns_all_configured(self, client):
        response = client.get("/api/v1/machines")
        assert response.status_code == 200
        data = response.json()
        ids = {m["id"] for m in data}
        assert {"AC-201", "HX-202", "CM-203", "AC-301", "HX-302", "CM-303"} == ids

    def test_machine_status_critical_with_active_alarm(self, client, db_session):
        """SHUTDOWN recommendation on an active alarm → critical tier."""
        seed_machine_data(db_session, ai_recommendation="SHUTDOWN")
        response = client.get("/api/v1/machines")
        machine = next(m for m in response.json() if m["id"] == "AC-201")
        assert machine["status"] == "critical"
        assert machine["top_alarm"] == "BEARING_FAULT"
        assert machine["rul_hours"] == 96.0
        assert machine["reliability"] == 84.0

    def test_status_observe_recommendation_yields_watch(self, client, db_session):
        """OBSERVE recommendation → watch (not critical, even with low health)."""
        now = datetime.now(UTC)
        db_session.add(
            SensorReading(
                machine_id="HX-202", timestamp=now,
                sensor_name="pressure_drop", value=1.3,
            )
        )
        db_session.add(
            MachineHealthScore(
                machine_id="HX-202",
                calculated_at=now,
                health_score=0.34,
                availability_score=0.9,
                reliability_score=0.9,
                condition_score=0.34,
                rul_hours=12.0,
                confidence=0.9,
                classification="Critical — Action Required",
            )
        )
        db_session.commit()
        # Seed an alarm + OBSERVE recommendation for HX-202.
        seed_machine_data(db_session, machine_id="HX-202", ai_recommendation="OBSERVE")
        machine = next(
            m for m in client.get("/api/v1/machines").json() if m["id"] == "HX-202"
        )
        assert machine["status"] == "watch"

    def test_status_planned_recommendation_yields_action(self, client, db_session):
        """PLANNED recommendation → action tier."""
        seed_machine_data(db_session, machine_id="CM-303", ai_recommendation="PLANNED")
        machine = next(
            m for m in client.get("/api/v1/machines").json() if m["id"] == "CM-303"
        )
        assert machine["status"] == "action"

    def test_status_active_alarm_no_recommendation_yields_watch(self, client, db_session):
        """Active alarm with no recommendation at all → watch (safe default)."""
        # seed_machine_data with no ai_recommendation: alarm exists, no recommendation set.
        seed_machine_data(db_session, machine_id="AC-301", ai_recommendation=None)
        # Ensure the DecisionLog has no recommendation (listener creates it with None).
        machine = next(
            m for m in client.get("/api/v1/machines").json() if m["id"] == "AC-301"
        )
        assert machine["status"] == "watch"

    def test_status_no_alarm_no_recommendation_yields_normal(self, client, db_session):
        """No active anomaly and no decision → normal."""
        now = datetime.now(UTC)
        db_session.add(
            SensorReading(
                machine_id="CM-203", timestamp=now,
                sensor_name="belt_tension", value=10.5,
            )
        )
        db_session.commit()
        machine = next(
            m for m in client.get("/api/v1/machines").json() if m["id"] == "CM-203"
        )
        assert machine["status"] == "normal"

    def test_status_latches_until_repair(self, client, db_session):
        """A flagged tier holds until a repair: an OBSERVE 'watch' does not
        rewind to normal when its alarm later resolves without a repair, and
        a real repair (downtime > 0) afterwards resets it."""
        now = datetime.now(UTC)
        db_session.add(
            SensorReading(
                machine_id="AC-201", timestamp=now,
                sensor_name="vibration_rms", value=3.0,
            )
        )
        anomaly = AnomalyLog(
            machine_id="AC-201", detected_at=now - timedelta(hours=2),
            anomaly_score=0.6, severity="WARNING", status="RESOLVED",
            fault_type="BEARING_FAULT",
        )
        db_session.add(anomaly)
        db_session.flush()
        alarm = AlarmState(
            anomaly_id=anomaly.id, machine_id="AC-201", level=1,
            status="NORMAL",  # alarm already resolved, not active
            created_at=now - timedelta(hours=2), last_updated=now - timedelta(hours=1),
        )
        db_session.add(alarm)
        db_session.flush()
        db_session.add(
            DecisionLog(
                alarm_id=alarm.id, machine_id="AC-201", action="APPROVE",
                ai_recommendation="OBSERVE", created_at=now - timedelta(hours=2),
            )
        )
        db_session.commit()
        # No repair yet → latched at watch even though the alarm is resolved.
        m = next(x for x in client.get("/api/v1/machines").json() if x["id"] == "AC-201")
        assert m["status"] == "watch"
        # A real repair (downtime > 0) after the decision resets the latch.
        db_session.add(
            MaintenanceLog(
                machine_id="AC-201", performed_at=now, downtime_minutes=45, cost_eur=3000.0,
            )
        )
        db_session.commit()
        m = next(x for x in client.get("/api/v1/machines").json() if x["id"] == "AC-201")
        assert m["status"] == "normal"

    def test_status_dispatch_technician_recommendation_yields_watch(self, client, db_session):
        """DISPATCH_TECHNICIAN recommendation → watch tier."""
        seed_machine_data(db_session, machine_id="HX-302", ai_recommendation="DISPATCH_TECHNICIAN")
        machine = next(
            m for m in client.get("/api/v1/machines").json() if m["id"] == "HX-302"
        )
        assert machine["status"] == "watch"

    def test_status_reduce_load_recommendation_yields_critical(self, client, db_session):
        """REDUCE_LOAD recommendation → critical tier."""
        seed_machine_data(db_session, machine_id="CM-303", ai_recommendation="REDUCE_LOAD")
        machine = next(
            m for m in client.get("/api/v1/machines").json() if m["id"] == "CM-303"
        )
        assert machine["status"] == "critical"

    def test_machine_offline_without_recent_readings(self, client):
        response = client.get("/api/v1/machines")
        machine = next(m for m in response.json() if m["id"] == "HX-302")
        assert machine["status"] == "offline"

    def test_machine_detail_includes_sensor_thresholds(self, client, db_session):
        seed_machine_data(db_session)
        response = client.get("/api/v1/machines/AC-201")
        assert response.status_code == 200
        data = response.json()
        vibration = next(s for s in data["sensors"] if s["sensor_name"] == "vibration_rms")
        assert vibration["warning_threshold"] == 4.5
        assert vibration["critical_threshold"] == 7.1
        assert vibration["value"] == 4.9
        assert data["active_faults"][0]["fault_type"] == "BEARING_FAULT"

    def test_machine_detail_unknown_returns_404(self, client):
        assert client.get("/api/v1/machines/NOPE-999").status_code == 404

    def test_sensor_series(self, client, db_session):
        seed_machine_data(db_session)
        response = client.get("/api/v1/machines/AC-201/sensors?minutes=60")
        assert response.status_code == 200
        series = response.json()["series"]
        assert "vibration_rms" in series
        assert series["vibration_rms"][0]["value"] == 4.9


class TestFleetEndpoints:
    def test_fleet_summary_counts(self, client, db_session):
        """SHUTDOWN recommendation on AC-201 → critical; remaining 5 machines offline."""
        seed_machine_data(db_session, ai_recommendation="SHUTDOWN")
        response = client.get("/api/v1/fleet/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 6
        assert data["critical"] == 1
        assert data["offline"] == 5
        assert data["active_alarms"] == 1

    def test_fleet_summary_new_tier_fields(self, client, db_session):
        """Fleet summary exposes online, watch, action, and warning == watch + action."""
        # AC-201: SHUTDOWN → critical  (online)
        seed_machine_data(db_session, machine_id="AC-201", ai_recommendation="SHUTDOWN")
        # AC-301: PLANNED → action     (online)
        seed_machine_data(db_session, machine_id="AC-301", ai_recommendation="PLANNED")
        # HX-302: OBSERVE → watch      (online)
        seed_machine_data(db_session, machine_id="HX-302", ai_recommendation="OBSERVE")
        # Remaining 3 machines have no readings → offline.
        response = client.get("/api/v1/fleet/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 6
        assert data["online"] == data["total"] - data["offline"], (
            "online must equal total − offline"
        )
        assert data["watch"] == 1
        assert data["action"] == 1
        assert data["critical"] == 1
        assert data["warning"] == data["watch"] + data["action"], (
            "warning is the backward-compat aggregate of watch + action"
        )
        assert data["offline"] == 3

    def test_fleet_health_trend(self, client, db_session):
        seed_machine_data(db_session)
        response = client.get("/api/v1/fleet/health-trend?hours=24")
        assert response.status_code == 200
        points = response.json()
        assert len(points) == 1
        assert points[0]["avg_health_score"] == 0.71


class TestAlarmsEndpoint:
    def test_active_alarms_listed_with_duration(self, client, db_session):
        seed_machine_data(db_session)
        response = client.get("/api/v1/alarms?active=true")
        assert response.status_code == 200
        alarms = response.json()
        assert len(alarms) == 1
        assert alarms[0]["machine_id"] == "AC-201"
        assert alarms[0]["severity"] == "CRITICAL"
        assert alarms[0]["duration_minutes"] >= 29


class TestDecisionsEndpoints:
    def test_pending_decision_exposes_scenarios(self, client, db_session):
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        assert decision is not None, "after_insert listener should create DecisionLog"
        decision.scenarios_presented = [
            {"scenario": "OBSERVE", "cost": 0.0, "is_recommended": False},
            {"scenario": "PLANNED", "cost": 15000.0, "is_recommended": True},
        ]
        decision.ai_recommendation = "PLANNED"
        db_session.commit()

        response = client.get("/api/v1/decisions/pending")
        assert response.status_code == 200
        pending = response.json()
        assert len(pending) == 1
        assert pending[0]["machine_id"] == "AC-201"
        assert pending[0]["severity"] == "CRITICAL"
        assert pending[0]["shap_values"] == {"vibration_rms_value": 0.42}
        assert {s["scenario"] for s in pending[0]["scenarios"]} == {"OBSERVE", "PLANNED"}

    def test_resolve_decision_approves_and_audits(self, client, db_session):
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        decision.ai_recommendation = "PLANNED"
        db_session.commit()

        response = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={"scenario_id": "PLANNED", "operator_role": "SUPERVISOR"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "APPROVE"
        assert data["chosen_scenario_id"] == "PLANNED"
        assert data["overridden"] is False
        audit_rows = db_session.query(DecisionAuditLog).all()
        assert len(audit_rows) == 1

    def test_resolve_invalid_scenario_rejected(self, client, db_session):
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        response = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={"scenario_id": "EXPLODE"},
        )
        assert response.status_code == 422

    def test_resolve_twice_returns_conflict(self, client, db_session):
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        first = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={"scenario_id": "OBSERVE", "operator_role": "SUPERVISOR", "operator_id": "HUMAN-SUP-1"},
        )
        assert first.status_code == 200
        second = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={"scenario_id": "OBSERVE", "operator_role": "SUPERVISOR", "operator_id": "HUMAN-SUP-1"},
        )
        assert second.status_code == 409

    def test_resolve_unknown_decision_404(self, client):
        response = client.post(
            "/api/v1/decisions/does-not-exist/resolve",
            json={"scenario_id": "OBSERVE", "operator_role": "SUPERVISOR", "operator_id": "HUMAN-SUP-1"},
        )
        assert response.status_code == 404


class TestDecisionRBAC:
    """Role-based access control for human operator resolutions.

    Bot identities (operator_id starting with "BOT-") must remain exempt so the
    closed-loop auto-approval loop never regresses.
    """

    def test_human_supervisor_shutdown_forbidden(self, client, db_session):
        """SUPERVISOR cannot execute SHUTDOWN — must get 403."""
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        response = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={
                "scenario_id": "SHUTDOWN",
                "operator_role": "SUPERVISOR",
                "operator_id": "HUMAN-SUP-1",
            },
        )
        assert response.status_code == 403

    def test_human_plant_manager_shutdown_allowed(self, client, db_session):
        """PLANT_MANAGER can execute SHUTDOWN — must succeed (200)."""
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        response = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={
                "scenario_id": "SHUTDOWN",
                "operator_role": "PLANT_MANAGER",
                "operator_id": "HUMAN-PLANT-1",
            },
        )
        assert response.status_code == 200

    def test_human_production_manager_reduce_load_allowed(self, client, db_session):
        """PRODUCTION_MANAGER can execute REDUCE_LOAD — must succeed (200)."""
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        response = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={
                "scenario_id": "REDUCE_LOAD",
                "operator_role": "PRODUCTION_MANAGER",
                "operator_id": "HUMAN-PMGR-1",
            },
        )
        assert response.status_code == 200

    def test_human_production_manager_shutdown_forbidden(self, client, db_session):
        """PRODUCTION_MANAGER cannot execute SHUTDOWN — must get 403."""
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        response = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={
                "scenario_id": "SHUTDOWN",
                "operator_role": "PRODUCTION_MANAGER",
                "operator_id": "HUMAN-PMGR-1",
            },
        )
        assert response.status_code == 403

    def test_bot_shutdown_allowed_no_regression(self, client, db_session):
        """BOT-* identities bypass RBAC and may resolve SHUTDOWN (auto-approval loop)."""
        alarm = seed_machine_data(db_session)
        decision = (
            db_session.query(DecisionLog).filter(DecisionLog.alarm_id == alarm.id).first()
        )
        response = client.post(
            f"/api/v1/decisions/{decision.id}/resolve",
            json={
                "scenario_id": "SHUTDOWN",
                "operator_role": "SUPERVISOR",
                "operator_id": "BOT-SUP-ALPHA",
            },
        )
        assert response.status_code == 200


class TestAuditEndpoint:
    def test_merged_audit_events(self, client, db_session):
        alarm = seed_machine_data(db_session)
        db_session.add(
            AlarmStateTransition(
                alarm_id=alarm.id,
                from_state="NORMAL",
                to_state="UNACKNOWLEDGED",
                operator_role="SYSTEM",
                timestamp=datetime.now(UTC),
            )
        )
        db_session.commit()

        response = client.get("/api/v1/audit/events")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        alarm_events = [e for e in data["events"] if e["category"] == "alarm"]
        assert alarm_events[0]["severity"] == "critical"
        assert alarm_events[0]["action"] == "ALARM_UNACKNOWLEDGED"

    def test_category_filter(self, client, db_session):
        alarm = seed_machine_data(db_session)
        db_session.add(
            AlarmStateTransition(
                alarm_id=alarm.id,
                from_state="NORMAL",
                to_state="ACKNOWLEDGED",
                timestamp=datetime.now(UTC),
            )
        )
        db_session.commit()
        response = client.get("/api/v1/audit/events?category=decision")
        assert all(e["category"] == "decision" for e in response.json()["events"])


class TestWorkOrdersEndpoint:
    def test_list_work_orders(self, client, db_session):
        db_session.add(
            WorkOrder(
                machine_id="CM-203",
                fault_type="BELT_SLIP",
                recommended_action="Replace belt tensioner",
                priority="HIGH",
                status="PENDING",
                estimated_cost_eur=1200.0,
                work_order_number="WO-CM-203-120000",
            )
        )
        db_session.commit()
        response = client.get("/api/v1/work-orders?machine_id=CM-203")
        assert response.status_code == 200
        orders = response.json()
        assert len(orders) == 1
        assert orders[0]["fault_type"] == "BELT_SLIP"
        assert orders[0]["priority"] == "HIGH"


class TestShiftReportsEndpoint:
    def test_list_shift_reports(self, client, db_session):
        now = datetime.now(UTC)
        db_session.add(
            ShiftReport(
                shift_start=now - timedelta(hours=8),
                shift_end=now,
                shift_type="A",
                report_data={"summary": "Quiet shift", "critical_count": 0},
            )
        )
        db_session.commit()
        response = client.get("/api/v1/shift-reports")
        assert response.status_code == 200
        reports = response.json()
        assert len(reports) == 1
        assert reports[0]["shift_type"] == "A"
        assert reports[0]["report_data"]["summary"] == "Quiet shift"


class TestMachineTimelineEndpoint:
    """GET /api/v1/machines/{machine_id}/timeline"""

    def test_timeline_unknown_machine_returns_404(self, client):
        response = client.get("/api/v1/machines/NOPE-999/timeline")
        assert response.status_code == 404

    def test_timeline_empty_when_no_decisions(self, client):
        """No decisions → count 0, empty events list."""
        response = client.get("/api/v1/machines/AC-201/timeline")
        assert response.status_code == 200
        data = response.json()
        assert data["machine_id"] == "AC-201"
        assert data["count"] == 0
        assert data["events"] == []
        assert data["repaired_at"] is None

    def test_timeline_events_newest_first(self, client, db_session):
        """Multiple decisions are returned newest first."""
        now = datetime.now(UTC)
        older = now - timedelta(hours=3)
        newer = now - timedelta(hours=1)

        # Two decisions without an intervening repair.
        db_session.add(
            DecisionLog(
                machine_id="AC-201",
                action="APPROVE",
                ai_recommendation="OBSERVE",
                decided_by="BOT-OPR",
                created_at=older,
            )
        )
        db_session.add(
            DecisionLog(
                machine_id="AC-201",
                action="APPROVE",
                ai_recommendation="PLANNED",
                decided_by="HUMAN-OP-1",
                created_at=newer,
            )
        )
        db_session.commit()

        response = client.get("/api/v1/machines/AC-201/timeline")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        # Newest first: PLANNED before OBSERVE.
        assert data["events"][0]["recommendation"] == "PLANNED"
        assert data["events"][0]["tier"] == "action"
        assert data["events"][1]["recommendation"] == "OBSERVE"
        assert data["events"][1]["tier"] == "watch"

    def test_timeline_excludes_events_before_last_repair(self, client, db_session):
        """Only events after the last real repair (downtime_minutes > 0) are returned."""
        now = datetime.now(UTC)
        repair_time = now - timedelta(hours=2)

        # Decision BEFORE the repair — must be excluded.
        db_session.add(
            DecisionLog(
                machine_id="AC-201",
                action="APPROVE",
                ai_recommendation="SHUTDOWN",
                decided_by="BOT-MGR",
                created_at=now - timedelta(hours=4),
            )
        )
        # Real repair.
        db_session.add(
            MaintenanceLog(
                machine_id="AC-201",
                performed_at=repair_time,
                downtime_minutes=60,
                cost_eur=5000.0,
            )
        )
        # Decision AFTER the repair — must be included.
        db_session.add(
            DecisionLog(
                machine_id="AC-201",
                action="PENDING",
                ai_recommendation="OBSERVE",
                decided_by=None,
                created_at=now - timedelta(hours=1),
            )
        )
        db_session.commit()

        response = client.get("/api/v1/machines/AC-201/timeline")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["events"][0]["recommendation"] == "OBSERVE"
        assert data["repaired_at"] is not None

    def test_timeline_tier_mapping(self, client, db_session):
        """Tier is derived correctly for all recommendation codes."""
        now = datetime.now(UTC)
        cases = [
            ("OBSERVE",            "watch"),
            ("DISPATCH_TECHNICIAN","watch"),
            ("PLANNED",            "action"),
            ("REDUCE_LOAD",        "critical"),
            ("SHUTDOWN",           "critical"),
        ]
        for idx, (rec, _) in enumerate(cases):
            db_session.add(
                DecisionLog(
                    machine_id="AC-201",
                    action="APPROVE",
                    ai_recommendation=rec,
                    created_at=now - timedelta(hours=len(cases) - idx),
                )
            )
        db_session.commit()

        response = client.get("/api/v1/machines/AC-201/timeline")
        assert response.status_code == 200
        events = response.json()["events"]
        # newest-first: reverse of insertion order
        actual = {e["recommendation"]: e["tier"] for e in events}
        for rec, expected_tier in cases:
            assert actual[rec] == expected_tier, f"{rec} → expected {expected_tier}, got {actual[rec]}"

    def test_timeline_inspection_without_downtime_does_not_reset_boundary(
        self, client, db_session
    ):
        """A MaintenanceLog with downtime_minutes=0 (inspection, no stop) must NOT
        reset the 'since last repair' boundary."""
        now = datetime.now(UTC)

        # Old decision.
        db_session.add(
            DecisionLog(
                machine_id="CM-203",
                action="APPROVE",
                ai_recommendation="OBSERVE",
                created_at=now - timedelta(hours=5),
            )
        )
        # Inspection visit — downtime_minutes=0, not a repair.
        db_session.add(
            MaintenanceLog(
                machine_id="CM-203",
                performed_at=now - timedelta(hours=3),
                downtime_minutes=0,
                cost_eur=0.0,
            )
        )
        # Newer decision.
        db_session.add(
            DecisionLog(
                machine_id="CM-203",
                action="APPROVE",
                ai_recommendation="PLANNED",
                created_at=now - timedelta(hours=1),
            )
        )
        db_session.commit()

        response = client.get("/api/v1/machines/CM-203/timeline")
        assert response.status_code == 200
        data = response.json()
        # Both decisions are included because the inspection does not reset the boundary.
        assert data["count"] == 2
        assert data["repaired_at"] is None
