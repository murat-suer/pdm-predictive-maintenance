"""
Unit tests for src.decision.decision_resolution (Phase 2D - Pipeline & Integration Layer).

Tests decision resolution helpers that:
- resolve_decision_plan(): Apply selected scenario
- Alarm state transitions (SHELVED, OUT_OF_SERVICE)
- DecisionAuditLog creation
- Override detection (AI recommendation vs human choice)
- Response time calculation
- Work order ID generation (WO-{machine_id}-{HHMMSS})

NOTE: These tests will FAIL until src/decision/decision_resolution.py is implemented.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy session."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.query = MagicMock()
    session.flush = MagicMock()
    return session


@pytest.fixture
def sample_alarm():
    """Sample AlarmState DB model instance."""
    alarm = MagicMock()
    alarm.id = 1
    alarm.anomaly_id = 42
    alarm.machine_id = "AC-201"
    alarm.status = "UNACKNOWLEDGED"
    alarm.level = 2
    alarm.created_at = datetime.now(UTC) - timedelta(minutes=5)
    alarm.last_updated = datetime.now(UTC)
    alarm.work_order_id = None
    alarm.shelved_at = None
    alarm.shelved_until = None
    alarm.oos_at = None
    return alarm


@pytest.fixture
def sample_decision():
    """Sample DecisionLog DB model instance."""
    decision = MagicMock()
    decision.id = "dec-001-uuid"
    decision.alarm_id = 1
    decision.machine_id = "AC-201"
    decision.action = "PENDING"
    decision.scenario_id = None
    decision.chosen_scenario_id = None
    decision.operator_role = None
    decision.decided_by = None
    decision.decided_at = None
    decision.response_time_s = None
    decision.ai_recommendation = "PLANNED"
    decision.overridden = False
    decision.escalation_level = 1
    decision.auto_approved = False
    decision.resolution_source = None
    decision.scenarios_presented = [
        {"scenario": "OBSERVE", "cost": 0.0, "is_recommended": False},
        {"scenario": "PLANNED", "cost": 25000.0, "is_recommended": True},
        {"scenario": "SHUTDOWN", "cost": 75000.0, "is_recommended": False},
    ]
    decision.created_at = datetime.now(UTC) - timedelta(minutes=3)
    return decision


@pytest.fixture
def sample_decision_with_override():
    """Sample DecisionLog where operator overrode AI recommendation."""
    decision = MagicMock()
    decision.id = "dec-002-uuid"
    decision.alarm_id = 2
    decision.machine_id = "HX-202"
    decision.action = "APPROVE"
    decision.scenario_id = "SHUTDOWN"
    decision.chosen_scenario_id = "SHUTDOWN"
    decision.operator_role = "MANAGER"
    decision.decided_by = "operator_john"
    decision.decided_at = datetime.now(UTC) - timedelta(minutes=1)
    decision.response_time_s = 120
    decision.ai_recommendation = "PLANNED"  # AI recommended PLANNED, human chose SHUTDOWN
    decision.overridden = True
    decision.escalation_level = 2
    decision.auto_approved = False
    decision.resolution_source = "HUMAN"
    decision.scenarios_presented = [
        {"scenario": "OBSERVE", "cost": 0.0, "is_recommended": False},
        {"scenario": "PLANNED", "cost": 25000.0, "is_recommended": True},
        {"scenario": "SHUTDOWN", "cost": 75000.0, "is_recommended": False},
    ]
    decision.created_at = datetime.now(UTC) - timedelta(minutes=3)
    return decision


# ---------------------------------------------------------------------------
# TestResolveDecisionPlan - Basic Functionality
# ---------------------------------------------------------------------------
class TestResolveDecisionPlanBasic:
    """Test resolve_decision_plan() basic functionality."""

    def test_resolve_returns_result(self, mock_db_session, sample_alarm, sample_decision):
        """resolve_decision_plan should return a result object."""
        from src.decision.decision_resolution import resolve_decision_plan
        result = resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert result is not None

    def test_resolve_sets_decision_action(self, mock_db_session, sample_alarm, sample_decision):
        """resolve_decision_plan should set decision action to APPROVE."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_decision.action == "APPROVE"

    def test_resolve_sets_chosen_scenario(self, mock_db_session, sample_alarm, sample_decision):
        """resolve_decision_plan should record chosen_scenario_id."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="OBSERVE",
            operator_role="OPERATOR",
            operator_id="operator_ali",
        )
        assert sample_decision.chosen_scenario_id == "OBSERVE"

    def test_resolve_sets_decided_at(self, mock_db_session, sample_alarm, sample_decision):
        """resolve_decision_plan should set decided_at timestamp."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_decision.decided_at is not None

    def test_resolve_sets_decided_by(self, mock_db_session, sample_alarm, sample_decision):
        """resolve_decision_plan should record who decided."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        assert sample_decision.decided_by == "manager_ayse"

    def test_resolve_commits_to_db(self, mock_db_session, sample_alarm, sample_decision):
        """resolve_decision_plan should commit changes to database."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        mock_db_session.commit.assert_called()


# ---------------------------------------------------------------------------
# TestScenarioSelection
# ---------------------------------------------------------------------------
class TestScenarioSelection:
    """Test scenario selection: AI recommended vs overridden."""

    def test_ai_recommended_scenario_selected(self, mock_db_session, sample_alarm, sample_decision):
        """When operator selects AI recommendation, overridden should be False."""
        from src.decision.decision_resolution import resolve_decision_plan
        # AI recommended PLANNED, operator also selects PLANNED
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",  # Same as ai_recommendation
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_decision.overridden is False

    def test_operator_overrides_ai_recommendation(self, mock_db_session, sample_alarm, sample_decision):
        """When operator selects different scenario, overridden should be True."""
        from src.decision.decision_resolution import resolve_decision_plan
        # AI recommended PLANNED, operator selects SHUTDOWN
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="SHUTDOWN",  # Different from ai_recommendation
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        assert sample_decision.overridden is True

    def test_observe_always_valid_choice(self, mock_db_session, sample_alarm, sample_decision):
        """OBSERVE should always be a valid choice (Murat's rule #1)."""
        from src.decision.decision_resolution import resolve_decision_plan
        result = resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="OBSERVE",
            operator_role="OPERATOR",
            operator_id="operator_mehmet",
        )
        assert result is not None
        assert sample_decision.chosen_scenario_id == "OBSERVE"


# ---------------------------------------------------------------------------
# TestAlarmStateTransition
# ---------------------------------------------------------------------------
class TestAlarmStateTransition:
    """Test alarm state transitions based on selected scenario."""

    def test_controlled_shutdown_transitions_to_out_of_service(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """SHUTDOWN scenario should transition alarm to OUT_OF_SERVICE."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="SHUTDOWN",
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        assert sample_alarm.status == "OUT_OF_SERVICE"

    def test_shutdown_sets_work_order_id(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """SHUTDOWN should generate and set work_order_id on alarm."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="SHUTDOWN",
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        assert sample_alarm.work_order_id is not None
        assert sample_alarm.oos_at is not None

    def test_planned_transitions_to_shelved(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """PLANNED scenario should transition alarm to SHELVED."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_alarm.status == "SHELVED"

    def test_observe_keeps_alarm_active(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """OBSERVE should keep alarm in ACKNOWLEDGED state (not SHELVED/OOS)."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="OBSERVE",
            operator_role="OPERATOR",
            operator_id="operator_mehmet",
        )
        assert sample_alarm.status in ("ACKNOWLEDGED", "UNACKNOWLEDGED")

    def test_reduce_load_keeps_alarm_active(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """REDUCE_LOAD should keep alarm in ACKNOWLEDGED state."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="REDUCE_LOAD",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_alarm.status in ("ACKNOWLEDGED", "UNACKNOWLEDGED")

    def test_shutdown_sets_oos_timestamp(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """SHUTDOWN should set oos_at timestamp."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="SHUTDOWN",
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        assert sample_alarm.oos_at is not None


# ---------------------------------------------------------------------------
# TestDecisionAuditLogCreation
# ---------------------------------------------------------------------------
class TestDecisionAuditLogCreation:
    """Test DecisionAuditLog record creation."""

    def test_audit_log_created_on_resolution(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Resolving a decision should create a DecisionAuditLog entry."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        # Should have added an audit log
        mock_db_session.add.assert_called()

    def test_audit_log_contains_decision_id(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """AuditLog should reference the decision_id."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        audit_logs = [obj for obj in added_objects if hasattr(obj, 'decision_id')]
        assert len(audit_logs) >= 1
        assert audit_logs[0].decision_id == sample_decision.id

    def test_audit_log_contains_scenario_id(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """AuditLog should record which scenario was selected."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="SHUTDOWN",
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        audit_logs = [obj for obj in added_objects if hasattr(obj, 'scenario_id')]
        assert len(audit_logs) >= 1
        assert audit_logs[0].scenario_id == "SHUTDOWN"

    def test_audit_log_contains_operator_role(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """AuditLog should record the operator's role."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        audit_logs = [obj for obj in added_objects if hasattr(obj, 'operator_role')]
        assert len(audit_logs) >= 1
        assert audit_logs[0].operator_role == "SUPERVISOR"


# ---------------------------------------------------------------------------
# TestOverrideDetection
# ---------------------------------------------------------------------------
class TestOverrideDetection:
    """Test override detection (AI recommendation vs human choice)."""

    def test_no_override_when_matching_ai(self, mock_db_session, sample_alarm, sample_decision):
        """No override when selected == ai_recommendation."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_decision.overridden is False

    def test_override_detected_when_different_from_ai(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Override detected when selected != ai_recommendation."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="SHUTDOWN",
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        assert sample_decision.overridden is True

    def test_override_audit_log_flag_set(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Override should be flagged in audit log."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="OBSERVE",  # Different from PLANNED
            operator_role="OPERATOR",
            operator_id="operator_mehmet",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        audit_logs = [obj for obj in added_objects if hasattr(obj, 'overridden')]
        assert len(audit_logs) >= 1
        assert audit_logs[0].overridden is True

    def test_no_override_audit_log_flag_when_matching(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """No override flag in audit log when matching AI recommendation."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",  # Same as ai_recommendation
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        audit_logs = [obj for obj in added_objects if hasattr(obj, 'overridden')]
        assert len(audit_logs) >= 1
        assert audit_logs[0].overridden is False

    def test_override_with_pre_existing_override_decision(
        self, mock_db_session, sample_alarm, sample_decision_with_override
    ):
        """Pre-existing override decision should maintain override flag."""
        from src.decision.decision_resolution import is_override
        decision = sample_decision_with_override
        assert is_override(decision) is True


# ---------------------------------------------------------------------------
# TestResponseTimeCalculation
# ---------------------------------------------------------------------------
class TestResponseTimeCalculation:
    """Test response time calculation."""

    def test_response_time_calculated_in_seconds(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Response time should be calculated in seconds."""
        from src.decision.decision_resolution import resolve_decision_plan
        # Set known timestamps
        sample_decision.created_at = datetime.now(UTC) - timedelta(seconds=90)

        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_decision.response_time_s is not None
        assert sample_decision.response_time_s >= 85  # ~90 seconds

    def test_response_time_zero_for_instant_decision(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Response time should be ~0 for instant decisions."""
        from src.decision.decision_resolution import resolve_decision_plan
        now = datetime.now(UTC)
        sample_decision.created_at = now

        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="OBSERVE",
            operator_role="OPERATOR",
            operator_id="operator_ali",
        )
        assert sample_decision.response_time_s is not None
        assert sample_decision.response_time_s >= 0

    def test_response_time_helper_function(self):
        """Test calculate_response_time helper function."""
        from src.decision.decision_resolution import calculate_response_time
        created = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
        decided = datetime(2026, 6, 10, 12, 2, 30, tzinfo=UTC)
        result = calculate_response_time(created, decided)
        assert result == 150  # 2 minutes 30 seconds = 150 seconds

    def test_response_time_negative_returns_zero(self):
        """Negative response time (clock skew) should return 0."""
        from src.decision.decision_resolution import calculate_response_time
        created = datetime(2026, 6, 10, 12, 5, 0, tzinfo=UTC)
        decided = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
        result = calculate_response_time(created, decided)
        assert result == 0


# ---------------------------------------------------------------------------
# TestWorkOrderIDGeneration
# ---------------------------------------------------------------------------
class TestWorkOrderIDGeneration:
    """Test work order ID generation: WO-{machine_id}-{HHMMSS}."""

    def test_work_order_id_format(self, mock_db_session, sample_alarm, sample_decision):
        """Work order ID should follow format WO-{machine_id}-{HHMMSS}."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="SHUTDOWN",
            operator_role="MANAGER",
            operator_id="manager_ayse",
        )
        wo_id = sample_alarm.work_order_id
        assert wo_id is not None
        assert wo_id.startswith("WO-AC-201-")

    def test_work_order_id_contains_machine_id(self):
        """Work order ID should contain the machine_id."""
        from src.decision.decision_resolution import generate_work_order_id
        wo_id = generate_work_order_id("HX-202")
        assert "HX-202" in wo_id

    def test_work_order_id_contains_timestamp(self):
        """Work order ID should contain HHMMSS timestamp component."""
        from src.decision.decision_resolution import generate_work_order_id
        wo_id = generate_work_order_id("AC-201")
        # Should have format WO-AC-201-HHMMSS
        parts = wo_id.split("-")
        assert len(parts) >= 4  # WO, AC, 201, HHMMSS
        # Last part should be 6 digits (HHMMSS)
        time_part = parts[-1]
        assert len(time_part) == 6
        assert time_part.isdigit()

    def test_work_order_id_prefix(self):
        """Work order ID should start with 'WO-'."""
        from src.decision.decision_resolution import generate_work_order_id
        wo_id = generate_work_order_id("CM-303")
        assert wo_id.startswith("WO-")

    def test_work_order_id_unique_per_second(self):
        """Two calls in same second should produce same ID (deterministic)."""
        from src.decision.decision_resolution import generate_work_order_id
        wo1 = generate_work_order_id("AC-201")
        wo2 = generate_work_order_id("AC-201")
        # Same machine, same second → same ID
        assert wo1 == wo2

    def test_work_order_id_different_machines(self):
        """Different machines should produce different work order IDs."""
        from src.decision.decision_resolution import generate_work_order_id
        wo1 = generate_work_order_id("AC-201")
        wo2 = generate_work_order_id("HX-202")
        assert wo1 != wo2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_scenario_id_raises_error(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Invalid scenario_id should raise ValueError."""
        from src.decision.decision_resolution import resolve_decision_plan
        with pytest.raises((ValueError, KeyError)):
            resolve_decision_plan(
                db=mock_db_session,
                alarm=sample_alarm,
                decision=sample_decision,
                selected_scenario_id="INVALID_SCENARIO",
                operator_role="OPERATOR",
                operator_id="operator_ali",
            )

    def test_missing_alarm_raises_error(self, mock_db_session, sample_decision):
        """Missing alarm (None) should raise ValueError."""
        from src.decision.decision_resolution import resolve_decision_plan
        with pytest.raises((ValueError, AttributeError)):
            resolve_decision_plan(
                db=mock_db_session,
                alarm=None,
                decision=sample_decision,
                selected_scenario_id="PLANNED",
                operator_role="SUPERVISOR",
                operator_id="operator_ali",
            )

    def test_already_resolved_decision_skipped(
        self, mock_db_session, sample_alarm
    ):
        """Already resolved decision should be skipped (idempotent)."""
        from src.decision.decision_resolution import resolve_decision_plan
        decision = MagicMock()
        decision.id = "dec-resolved"
        decision.action = "APPROVE"  # Already resolved
        decision.chosen_scenario_id = "PLANNED"

        result = resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=decision,
            selected_scenario_id="OBSERVE",
            operator_role="OPERATOR",
            operator_id="operator_ali",
        )
        # Should return early or indicate already resolved
        # Should not create duplicate audit log
        assert result is not None

    def test_none_decision_raises_error(self, mock_db_session, sample_alarm):
        """None decision should raise ValueError."""
        from src.decision.decision_resolution import resolve_decision_plan
        with pytest.raises((ValueError, AttributeError)):
            resolve_decision_plan(
                db=mock_db_session,
                alarm=sample_alarm,
                decision=None,
                selected_scenario_id="PLANNED",
                operator_role="SUPERVISOR",
                operator_id="operator_ali",
            )

    def test_empty_operator_id_still_works(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Empty operator_id should still resolve (auto-approved)."""
        from src.decision.decision_resolution import resolve_decision_plan
        result = resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="OBSERVE",
            operator_role="OPERATOR",
            operator_id="",
        )
        assert result is not None

    def test_db_failure_triggers_rollback(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """DB failure during resolution should trigger rollback."""
        from src.decision.decision_resolution import resolve_decision_plan
        mock_db_session.commit.side_effect = Exception("DB down")

        with pytest.raises(Exception):
            resolve_decision_plan(
                db=mock_db_session,
                alarm=sample_alarm,
                decision=sample_decision,
                selected_scenario_id="PLANNED",
                operator_role="SUPERVISOR",
                operator_id="operator_ali",
            )
        mock_db_session.rollback.assert_called()

    def test_resolution_source_set_to_human(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Human resolution should set resolution_source to 'HUMAN'."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role="SUPERVISOR",
            operator_id="operator_ali",
        )
        assert sample_decision.resolution_source == "HUMAN"

    def test_auto_approval_sets_resolution_source(
        self, mock_db_session, sample_alarm, sample_decision
    ):
        """Auto-approval should set resolution_source to 'AUTO'."""
        from src.decision.decision_resolution import resolve_decision_plan
        resolve_decision_plan(
            db=mock_db_session,
            alarm=sample_alarm,
            decision=sample_decision,
            selected_scenario_id="PLANNED",
            operator_role=None,
            operator_id=None,
            auto_approve=True,
        )
        assert sample_decision.resolution_source == "AUTO"
        assert sample_decision.auto_approved is True
