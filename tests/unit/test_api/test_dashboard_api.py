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


def seed_machine_data(db, machine_id="AC-201"):
    """Insert one healthy reading, health score, anomaly and alarm."""
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
        seed_machine_data(db_session)
        response = client.get("/api/v1/machines")
        machine = next(m for m in response.json() if m["id"] == "AC-201")
        assert machine["status"] == "critical"
        assert machine["top_alarm"] == "BEARING_FAULT"
        assert machine["rul_hours"] == 96.0
        assert machine["reliability"] == 84.0

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
        seed_machine_data(db_session)
        response = client.get("/api/v1/fleet/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 6
        assert data["critical"] == 1
        assert data["offline"] == 5
        assert data["active_alarms"] == 1

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
            f"/api/v1/decisions/{decision.id}/resolve", json={"scenario_id": "OBSERVE"}
        )
        assert first.status_code == 200
        second = client.post(
            f"/api/v1/decisions/{decision.id}/resolve", json={"scenario_id": "OBSERVE"}
        )
        assert second.status_code == 409

    def test_resolve_unknown_decision_404(self, client):
        response = client.post(
            "/api/v1/decisions/does-not-exist/resolve", json={"scenario_id": "OBSERVE"}
        )
        assert response.status_code == 404


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
