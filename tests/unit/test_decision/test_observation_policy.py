"""Tests for the repeated-OBSERVE escalation policy (EEMUA 191 guard)."""
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import AnomalyLog, DecisionLog, MaintenanceLog
from src.decision.observation_policy import (
    DISPATCH_COST_EUR,
    DISPATCH_SCENARIO,
    apply_observe_escalation,
    dispatched_since_repair,
    fault_is_identified,
    observe_streak,
)


def make_scenarios(recommended="OBSERVE"):
    scenarios = [
        {"scenario": "OBSERVE", "cost": 0.0, "expected_cost": 850.0,
         "failure_probability": 0.02, "is_recommended": False},
        {"scenario": "PLANNED", "cost": 3370.0, "expected_cost": 3010.0,
         "failure_probability": 0.001, "is_recommended": False},
        {"scenario": "REDUCE_LOAD", "cost": 144.0, "expected_cost": 6918.0,
         "failure_probability": 0.01, "is_recommended": False},
        {"scenario": "SHUTDOWN", "cost": 8229.0, "expected_cost": 8229.0,
         "failure_probability": 0.0, "is_recommended": False},
    ]
    for s in scenarios:
        s["is_recommended"] = s["scenario"] == recommended
    return scenarios


class TestApplyObserveEscalation:
    def test_streak_zero_changes_nothing(self):
        scenarios, rec = apply_observe_escalation(make_scenarios(), "OBSERVE", 0, False)
        assert rec == "OBSERVE"
        assert [s["scenario"] for s in scenarios] == [
            "OBSERVE", "PLANNED", "REDUCE_LOAD", "SHUTDOWN"
        ]

    def test_second_decision_unidentified_recommends_dispatch(self):
        scenarios, rec = apply_observe_escalation(make_scenarios(), "OBSERVE", 1, False)
        ids = [s["scenario"] for s in scenarios]
        assert DISPATCH_SCENARIO in ids
        assert rec == DISPATCH_SCENARIO
        recommended = [s for s in scenarios if s["is_recommended"]]
        assert len(recommended) == 1
        assert recommended[0]["scenario"] == DISPATCH_SCENARIO
        assert recommended[0]["cost"] == DISPATCH_COST_EUR

    def test_identified_fault_keeps_observe_recommendation(self):
        scenarios, rec = apply_observe_escalation(make_scenarios(), "OBSERVE", 1, True)
        # dispatch is offered but the cost-optimal recommendation stands
        assert DISPATCH_SCENARIO in [s["scenario"] for s in scenarios]
        assert rec == "OBSERVE"

    def test_planned_overridden_to_dispatch_for_unidentified_fault(self):
        # When the cost model picks PLANNED but the fault is unclassified,
        # scheduling maintenance for an unknown problem is worse than a
        # €150 on-line inspection — dispatch must take priority.
        scenarios, rec = apply_observe_escalation(
            make_scenarios(recommended="PLANNED"), "PLANNED", 2, False
        )
        assert rec == DISPATCH_SCENARIO

    def test_planned_recommendation_kept_for_identified_fault(self):
        # An identified fault with a streak: the cost model knows what it
        # is recommending — PLANNED is correct, dispatch is merely offered.
        scenarios, rec = apply_observe_escalation(
            make_scenarios(recommended="PLANNED"), "PLANNED", 2, True
        )
        assert rec == "PLANNED"

    def test_shutdown_recommendation_is_not_overridden(self):
        # SHUTDOWN means imminent failure — a dispatch inspection would be
        # too slow.  The stronger signal must win.
        scenarios, rec = apply_observe_escalation(
            make_scenarios(recommended="SHUTDOWN"), "SHUTDOWN", 2, False
        )
        assert rec == "SHUTDOWN"

    def test_fourth_decision_drops_observe(self):
        scenarios, rec = apply_observe_escalation(make_scenarios(), "OBSERVE", 3, False)
        ids = [s["scenario"] for s in scenarios]
        assert "OBSERVE" not in ids
        assert DISPATCH_SCENARIO in ids
        assert rec == DISPATCH_SCENARIO

    def test_observe_dropped_identified_recommends_planned(self):
        # An identified fault watched to the limit is already diagnosed —
        # schedule the repair (PLANNED) rather than dispatch a technician to
        # re-diagnose what we already know.
        scenarios, rec = apply_observe_escalation(make_scenarios(), "OBSERVE", 3, True)
        assert "OBSERVE" not in [s["scenario"] for s in scenarios]
        assert rec == "PLANNED"

    def test_observe_dropped_unidentified_recommends_dispatch(self):
        # Still unidentified after three watches → send a technician to find
        # out what it is (dispatch), not schedule a blind overhaul.
        scenarios, rec = apply_observe_escalation(make_scenarios(), "OBSERVE", 3, False)
        assert "OBSERVE" not in [s["scenario"] for s in scenarios]
        assert rec == DISPATCH_SCENARIO

    def test_exactly_one_recommended_flag(self):
        for streak in (0, 1, 2, 3, 5):
            scenarios, _ = apply_observe_escalation(make_scenarios(), "OBSERVE", streak, False)
            assert sum(1 for s in scenarios if s["is_recommended"]) == 1


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    DecisionLog.__table__.create(engine, checkfirst=True)
    AnomalyLog.__table__.create(engine, checkfirst=True)
    MaintenanceLog.__table__.create(engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def add_decision(db, machine_id, chosen, minutes_ago):
    db.add(
        DecisionLog(
            machine_id=machine_id,
            action="APPROVE",
            chosen_scenario_id=chosen,
            decided_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
            created_at=datetime.now(UTC) - timedelta(minutes=minutes_ago + 1),
        )
    )
    db.commit()


class TestObserveStreak:
    def test_counts_consecutive_observes(self, db):
        add_decision(db, "AC-201", "OBSERVE", 30)
        add_decision(db, "AC-201", "OBSERVE", 20)
        assert observe_streak(db, "AC-201") == 2

    def test_intervention_resets_streak(self, db):
        add_decision(db, "AC-201", "OBSERVE", 40)
        add_decision(db, "AC-201", "PLANNED", 30)
        add_decision(db, "AC-201", "OBSERVE", 20)
        assert observe_streak(db, "AC-201") == 1

    def test_dispatch_does_not_reset_streak(self, db):
        add_decision(db, "AC-201", "OBSERVE", 40)
        add_decision(db, "AC-201", DISPATCH_SCENARIO, 30)
        add_decision(db, "AC-201", "OBSERVE", 20)
        assert observe_streak(db, "AC-201") == 2

    def test_other_machine_does_not_count(self, db):
        add_decision(db, "HX-202", "OBSERVE", 30)
        assert observe_streak(db, "AC-201") == 0


class TestDispatchedSinceRepair:
    def test_no_dispatch_is_false(self, db):
        add_decision(db, "AC-201", "OBSERVE", 30)
        assert dispatched_since_repair(db, "AC-201") is False

    def test_dispatch_makes_it_identified(self, db):
        add_decision(db, "AC-201", DISPATCH_SCENARIO, 20)
        assert dispatched_since_repair(db, "AC-201") is True

    def test_repair_after_dispatch_resets(self, db):
        add_decision(db, "AC-201", DISPATCH_SCENARIO, 60)
        db.add(
            MaintenanceLog(
                machine_id="AC-201",
                performed_at=datetime.now(UTC) - timedelta(minutes=30),
                downtime_minutes=45,
            )
        )
        db.commit()
        assert dispatched_since_repair(db, "AC-201") is False


class TestFaultIdentified:
    def test_unclassified_is_not_identified(self, db):
        anomaly = AnomalyLog(
            machine_id="AC-201",
            detected_at=datetime.now(UTC),
            anomaly_score=0.8,
            severity="WARNING",
            status="ACTIVE",
            fault_type="UNCLASSIFIED_ANOMALY",
        )
        db.add(anomaly)
        db.commit()
        assert fault_is_identified(db, anomaly.id) is False

    def test_named_fault_is_identified(self, db):
        anomaly = AnomalyLog(
            machine_id="AC-201",
            detected_at=datetime.now(UTC),
            anomaly_score=0.8,
            severity="WARNING",
            status="ACTIVE",
            fault_type="BEARING_FAULT",
        )
        db.add(anomaly)
        db.commit()
        assert fault_is_identified(db, anomaly.id) is True

    def test_missing_anomaly_is_not_identified(self, db):
        assert fault_is_identified(db, None) is False
        assert fault_is_identified(db, 99999) is False
