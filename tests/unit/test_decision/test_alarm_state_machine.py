"""
Unit tests for src.decision.alarm_state_machine

Tests AlarmStateMachine: ISA-18.2 compliant state transitions,
role hierarchy validation, shelve/suppress/out-of-service logic,
and audit trail creation.
"""

from datetime import datetime, timedelta

import pytest

from src.decision.alarm_state_machine import (
    MAX_SHELVE_DURATION_HOURS,
    MAX_SUPPRESS_DURATION_DAYS,
    AlarmState,
    AlarmStateMachine,
    Role,
    TransitionError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def machine():
    """Fresh alarm state machine in NORMAL state."""
    return AlarmStateMachine(alarm_id="ALM-001", machine_id="AC-001")


@pytest.fixture
def shelved_machine():
    """Machine in SHELVED state."""
    m = AlarmStateMachine(alarm_id="ALM-002", machine_id="AC-002")
    m.shelve(
        role=Role.SUPERVISOR,
        reason="Scheduled maintenance window",
        duration_hours=4.0,
    )
    return m


# ---------------------------------------------------------------------------
# TestISA182StateTransitions
# ---------------------------------------------------------------------------
class TestISA182StateTransitions:
    """Test all valid ISA-18.2 state transitions."""

    def test_normal_to_active(self, machine):
        """NORMAL → UNACKNOWLEDGED (alarm occurs)."""
        machine.transition_to(AlarmState.UNACKNOWLEDGED, role=Role.OPERATOR)
        assert machine.state == AlarmState.UNACKNOWLEDGED

    def test_unacknowledged_to_acknowledged(self, machine):
        """UNACKNOWLEDGED → ACKNOWLEDGED."""
        machine.transition_to(AlarmState.UNACKNOWLEDGED, role=Role.OPERATOR)
        machine.transition_to(AlarmState.ACKNOWLEDGED, role=Role.OPERATOR)
        assert machine.state == AlarmState.ACKNOWLEDGED

    def test_acknowledged_to_normal(self, machine):
        """ACKNOWLEDGED → NORMAL (alarm returns to normal)."""
        machine.transition_to(AlarmState.UNACKNOWLEDGED, role=Role.OPERATOR)
        machine.transition_to(AlarmState.ACKNOWLEDGED, role=Role.OPERATOR)
        machine.transition_to(AlarmState.NORMAL, role=Role.OPERATOR)
        assert machine.state == AlarmState.NORMAL

    def test_normal_to_shelved(self, machine):
        """NORMAL → SHELVED (preventive shelve)."""
        machine.shelve(
            role=Role.SUPERVISOR,
            reason="Maintenance in progress",
            duration_hours=2.0,
        )
        assert machine.state == AlarmState.SHELVED

    def test_normal_to_out_of_service(self, machine):
        """NORMAL → OUT_OF_SERVICE."""
        machine.set_out_of_service(
            role=Role.MANAGER,
            work_order_id="WO-2026-001",
        )
        assert machine.state == AlarmState.OUT_OF_SERVICE

    def test_normal_to_suppressed(self, machine):
        """NORMAL → SUPPRESSED (Manager only)."""
        machine.suppress(
            role=Role.MANAGER,
            reason="Design-level suppression",
        )
        assert machine.state == AlarmState.SUPPRESSED

    def test_shelved_to_normal(self, shelved_machine):
        """SHELVED → NORMAL (unshelve)."""
        shelved_machine.unshelve(role=Role.SUPERVISOR)
        assert shelved_machine.state == AlarmState.NORMAL

    def test_out_of_service_to_normal(self, machine):
        """OUT_OF_SERVICE → NORMAL (return to service)."""
        machine.set_out_of_service(
            role=Role.MANAGER,
            work_order_id="WO-2026-001",
        )
        machine.return_to_service(role=Role.MANAGER)
        assert machine.state == AlarmState.NORMAL


# ---------------------------------------------------------------------------
# TestRoleHierarchy
# ---------------------------------------------------------------------------
class TestRoleHierarchy:
    """Role hierarchy: Operator < Supervisor < Manager."""

    def test_role_ordering(self):
        assert Role.OPERATOR < Role.SUPERVISOR
        assert Role.SUPERVISOR < Role.MANAGER

    def test_operator_cannot_shelve(self, machine):
        """Operator should not be able to shelve alarms."""
        with pytest.raises((TransitionError, PermissionError)):
            machine.shelve(
                role=Role.OPERATOR,
                reason="test",
                duration_hours=2.0,
            )

    def test_supervisor_can_shelve(self, machine):
        """Supervisor can shelve."""
        machine.shelve(
            role=Role.SUPERVISOR,
            reason="test",
            duration_hours=2.0,
        )
        assert machine.state == AlarmState.SHELVED

    def test_operator_cannot_suppress(self, machine):
        """SUPPRESSED requires Manager role."""
        with pytest.raises((TransitionError, PermissionError)):
            machine.suppress(role=Role.OPERATOR, reason="test")

    def test_supervisor_cannot_suppress(self, machine):
        """SUPPRESSED requires Manager role - Supervisor not enough."""
        with pytest.raises((TransitionError, PermissionError)):
            machine.suppress(role=Role.SUPERVISOR, reason="test")

    def test_manager_can_suppress(self, machine):
        """Manager can suppress."""
        machine.suppress(role=Role.MANAGER, reason="Design suppression")
        assert machine.state == AlarmState.SUPPRESSED


# ---------------------------------------------------------------------------
# TestShelvedState
# ---------------------------------------------------------------------------
class TestShelvedState:
    """SHELVED state: reason mandatory, duration limits."""

    def test_shelve_requires_reason(self, machine):
        """Shelve without reason should fail."""
        with pytest.raises((ValueError, TransitionError)):
            machine.shelve(
                role=Role.SUPERVISOR,
                reason="",
                duration_hours=2.0,
            )

    def test_shelve_none_reason_rejected(self, machine):
        with pytest.raises((ValueError, TransitionError)):
            machine.shelve(
                role=Role.SUPERVISOR,
                reason=None,
                duration_hours=2.0,
            )

    def test_shelve_duration_limit(self, machine):
        """Shelve duration exceeding max should be rejected."""
        with pytest.raises((ValueError, TransitionError)):
            machine.shelve(
                role=Role.SUPERVISOR,
                reason="test",
                duration_hours=MAX_SHELVE_DURATION_HOURS + 1,
            )

    def test_shelve_at_max_duration(self, machine):
        """Exactly at max duration should be accepted."""
        machine.shelve(
            role=Role.SUPERVISOR,
            reason="Extended maintenance",
            duration_hours=MAX_SHELVE_DURATION_HOURS,
        )
        assert machine.state == AlarmState.SHELVED

    def test_shelve_expiry_auto_unshelve(self, machine):
        """After shelve duration expires, should auto-return to NORMAL."""
        machine.shelve(
            role=Role.SUPERVISOR,
            reason="Short maintenance",
            duration_hours=1.0,
        )
        # Simulate time passing beyond shelve duration
        future = datetime.utcnow() + timedelta(hours=2)
        machine.check_expiry(now=future)
        assert machine.state == AlarmState.NORMAL

    def test_shelve_not_expired_yet(self, machine):
        """Before expiry, should remain SHELVED."""
        machine.shelve(
            role=Role.SUPERVISOR,
            reason="Short maintenance",
            duration_hours=4.0,
        )
        future = datetime.utcnow() + timedelta(hours=1)
        machine.check_expiry(now=future)
        assert machine.state == AlarmState.SHELVED


# ---------------------------------------------------------------------------
# TestOutOfServiceState
# ---------------------------------------------------------------------------
class TestOutOfServiceState:
    """OUT_OF_SERVICE: work_order_id mandatory."""

    def test_oos_requires_work_order(self, machine):
        """OUT_OF_SERVICE without work_order_id should fail."""
        with pytest.raises((ValueError, TransitionError)):
            machine.set_out_of_service(
                role=Role.MANAGER,
                work_order_id="",
            )

    def test_oos_none_work_order_rejected(self, machine):
        with pytest.raises((ValueError, TransitionError)):
            machine.set_out_of_service(
                role=Role.MANAGER,
                work_order_id=None,
            )

    def test_oos_with_valid_work_order(self, machine):
        machine.set_out_of_service(
            role=Role.MANAGER,
            work_order_id="WO-2026-042",
        )
        assert machine.state == AlarmState.OUT_OF_SERVICE
        assert machine.work_order_id == "WO-2026-042"


# ---------------------------------------------------------------------------
# TestSuppressedState
# ---------------------------------------------------------------------------
class TestSuppressedState:
    """SUPPRESSED: Manager only, 30-day review."""

    def test_suppress_requires_manager(self, machine):
        """Only Manager can suppress."""
        with pytest.raises((TransitionError, PermissionError)):
            machine.suppress(role=Role.SUPERVISOR, reason="test")

    def test_suppress_30_day_review(self, machine):
        """Suppression should have a review date within 30 days."""
        machine.suppress(role=Role.MANAGER, reason="Design suppression")
        assert machine.review_date is not None
        # Review date should be within MAX_SUPPRESS_DURATION_DAYS
        days_until_review = (machine.review_date - datetime.utcnow()).days
        assert days_until_review <= MAX_SUPPRESS_DURATION_DAYS

    def test_suppress_requires_reason(self, machine):
        with pytest.raises((ValueError, TransitionError)):
            machine.suppress(role=Role.MANAGER, reason="")


# ---------------------------------------------------------------------------
# TestInvalidTransitions
# ---------------------------------------------------------------------------
class TestInvalidTransitions:
    """Invalid state transitions should be rejected."""

    def test_normal_to_acknowledged_invalid(self, machine):
        """Cannot ACK without first being UNACKNOWLEDGED."""
        with pytest.raises(TransitionError):
            machine.transition_to(AlarmState.ACKNOWLEDGED, role=Role.OPERATOR)

    def test_shelved_to_suppressed_invalid(self, shelved_machine):
        """Cannot suppress a shelved alarm."""
        with pytest.raises(TransitionError):
            shelved_machine.suppress(role=Role.MANAGER, reason="test")

    def test_out_of_service_to_shelved_invalid(self, machine):
        """Cannot shelve an out-of-service alarm."""
        machine.set_out_of_service(
            role=Role.MANAGER,
            work_order_id="WO-001",
        )
        with pytest.raises(TransitionError):
            machine.shelve(
                role=Role.SUPERVISOR,
                reason="test",
                duration_hours=2.0,
            )

    def test_double_shelve_rejected(self, shelved_machine):
        """Cannot shelve an already shelved alarm."""
        with pytest.raises(TransitionError):
            shelved_machine.shelve(
                role=Role.SUPERVISOR,
                reason="double shelve",
                duration_hours=1.0,
            )


# ---------------------------------------------------------------------------
# TestAuditTrail
# ---------------------------------------------------------------------------
class TestAuditTrail:
    """Every state transition should create an audit entry."""

    def test_transition_creates_audit_entry(self, machine):
        initial_count = len(machine.audit_trail)
        machine.transition_to(AlarmState.UNACKNOWLEDGED, role=Role.OPERATOR)
        assert len(machine.audit_trail) == initial_count + 1

    def test_audit_entry_has_required_fields(self, machine):
        machine.transition_to(AlarmState.UNACKNOWLEDGED, role=Role.OPERATOR)
        entry = machine.audit_trail[-1]
        assert entry.from_state == AlarmState.NORMAL
        assert entry.to_state == AlarmState.UNACKNOWLEDGED
        assert entry.role == Role.OPERATOR
        assert entry.timestamp is not None

    def test_shelve_audit_includes_reason(self, machine):
        machine.shelve(
            role=Role.SUPERVISOR,
            reason="Maintenance window",
            duration_hours=2.0,
        )
        entry = machine.audit_trail[-1]
        assert entry.reason == "Maintenance window"

    def test_oos_audit_includes_work_order(self, machine):
        machine.set_out_of_service(
            role=Role.MANAGER,
            work_order_id="WO-999",
        )
        entry = machine.audit_trail[-1]
        assert entry.work_order_id == "WO-999"

    def test_audit_trail_is_ordered(self, machine):
        """Audit trail should be in chronological order."""
        machine.transition_to(AlarmState.UNACKNOWLEDGED, role=Role.OPERATOR)
        machine.transition_to(AlarmState.ACKNOWLEDGED, role=Role.OPERATOR)
        machine.transition_to(AlarmState.NORMAL, role=Role.OPERATOR)
        timestamps = [e.timestamp for e in machine.audit_trail]
        assert timestamps == sorted(timestamps)

    def test_multiple_transitions_all_audited(self, machine):
        machine.transition_to(AlarmState.UNACKNOWLEDGED, role=Role.OPERATOR)
        machine.transition_to(AlarmState.ACKNOWLEDGED, role=Role.OPERATOR)
        machine.shelve(
            role=Role.SUPERVISOR,
            reason="test",
            duration_hours=1.0,
        )
        # At least 3 audit entries
        assert len(machine.audit_trail) >= 3
