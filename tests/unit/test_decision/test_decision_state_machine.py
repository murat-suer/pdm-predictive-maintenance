"""
Unit tests for src.decision.decision_state_machine (Phase 2B - Core Logic Layer)

Tests DecisionStateMachine: 7-state machine (HEALTHY, MONITORING, ALARMED,
MONITOR_30M, ESCALATED, RECOVERED, STABLE) with 30-minute watchdog,
anti-pattern guard (EEMUA 191: max 3 same option), and time-based checks.

NOTE: These tests will FAIL until decision_state_machine.py is implemented.
"""

from datetime import datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.decision_state_machine import (
    CONTROLLED_STOP_OPTION,
    HEALTHY_DURATION_MINUTES,
    MAX_OPTION_REPEATS,
    STABLE_DURATION_MINUTES,
    WATCHDOG_DURATION_MINUTES,
    AntiPatternViolation,
    DecisionState,
    DecisionStateMachine,
    InvalidTransitionError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def machine():
    """Fresh DecisionStateMachine in HEALTHY state."""
    return DecisionStateMachine(machine_id="AC-001")


@pytest.fixture
def alarmed_machine():
    """Machine in ALARMED state."""
    m = DecisionStateMachine(machine_id="AC-001")
    m.transition(DecisionState.MONITORING)
    m.transition(DecisionState.ALARMED)
    return m


@pytest.fixture
def monitor_30m_machine():
    """Machine in MONITOR_30M state with active watchdog."""
    m = DecisionStateMachine(machine_id="AC-001")
    m.transition(DecisionState.MONITORING)
    m.transition(DecisionState.ALARMED)
    m.transition(DecisionState.MONITOR_30M)
    return m


@pytest.fixture
def escalated_machine():
    """Machine in ESCALATED state."""
    m = DecisionStateMachine(machine_id="AC-001")
    m.transition(DecisionState.MONITORING)
    m.transition(DecisionState.ALARMED)
    m.transition(DecisionState.MONITOR_30M)
    m.transition(DecisionState.ESCALATED)
    return m


# ---------------------------------------------------------------------------
# TestValidStateTransitions
# ---------------------------------------------------------------------------
class TestValidStateTransitions:
    """Test all 7 valid state transitions in the lifecycle."""

    def test_healthy_to_monitoring(self, machine):
        """HEALTHY → MONITORING (anomaly detected)."""
        machine.transition(DecisionState.MONITORING)
        assert machine.state == DecisionState.MONITORING

    def test_monitoring_to_alarmed(self, machine):
        """MONITORING → ALARMED (threshold exceeded)."""
        machine.transition(DecisionState.MONITORING)
        machine.transition(DecisionState.ALARMED)
        assert machine.state == DecisionState.ALARMED

    def test_alarmed_to_monitor_30m(self, alarmed_machine):
        """ALARMED → MONITOR_30M (30-min watchdog starts)."""
        alarmed_machine.transition(DecisionState.MONITOR_30M)
        assert alarmed_machine.state == DecisionState.MONITOR_30M

    def test_monitor_30m_to_escalated(self, monitor_30m_machine):
        """MONITOR_30M → ESCALATED (watchdog expired, no recovery)."""
        monitor_30m_machine.transition(DecisionState.ESCALATED)
        assert monitor_30m_machine.state == DecisionState.ESCALATED

    def test_monitor_30m_to_recovered(self, monitor_30m_machine):
        """MONITOR_30M → RECOVERED (values returned to normal)."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        assert monitor_30m_machine.state == DecisionState.RECOVERED

    def test_recovered_to_stable(self, monitor_30m_machine):
        """RECOVERED → STABLE (sustained recovery)."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        monitor_30m_machine.transition(DecisionState.STABLE)
        assert monitor_30m_machine.state == DecisionState.STABLE

    def test_stable_to_healthy(self, monitor_30m_machine):
        """STABLE → HEALTHY (full recovery)."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        monitor_30m_machine.transition(DecisionState.STABLE)
        monitor_30m_machine.transition(DecisionState.HEALTHY)
        assert monitor_30m_machine.state == DecisionState.HEALTHY

    def test_full_lifecycle(self, machine):
        """Full lifecycle: HEALTHY → MONITORING → ALARMED → MONITOR_30M →
        RECOVERED → STABLE → HEALTHY."""
        machine.transition(DecisionState.MONITORING)
        machine.transition(DecisionState.ALARMED)
        machine.transition(DecisionState.MONITOR_30M)
        machine.transition(DecisionState.RECOVERED)
        machine.transition(DecisionState.STABLE)
        machine.transition(DecisionState.HEALTHY)
        assert machine.state == DecisionState.HEALTHY

    def test_escalated_to_recovered(self, escalated_machine):
        """ESCALATED → RECOVERED (after intervention)."""
        escalated_machine.transition(DecisionState.RECOVERED)
        assert escalated_machine.state == DecisionState.RECOVERED


# ---------------------------------------------------------------------------
# TestIllegalTransitionRejection
# ---------------------------------------------------------------------------
class TestIllegalTransitionRejection:
    """Invalid transitions should be rejected."""

    def test_healthy_to_alarmed_invalid(self, machine):
        """Cannot skip MONITORING to go directly to ALARMED."""
        with pytest.raises(InvalidTransitionError):
            machine.transition(DecisionState.ALARMED)

    def test_healthy_to_escalated_invalid(self, machine):
        """Cannot skip multiple states to ESCALATED."""
        with pytest.raises(InvalidTransitionError):
            machine.transition(DecisionState.ESCALATED)

    def test_monitoring_to_monitor_30m_invalid(self, machine):
        """Cannot go to MONITOR_30M without being ALARMED first."""
        machine.transition(DecisionState.MONITORING)
        with pytest.raises(InvalidTransitionError):
            machine.transition(DecisionState.MONITOR_30M)

    def test_healthy_to_recovered_invalid(self, machine):
        """Cannot go to RECOVERED from HEALTHY."""
        with pytest.raises(InvalidTransitionError):
            machine.transition(DecisionState.RECOVERED)

    def test_alarmed_to_healthy_invalid(self, alarmed_machine):
        """Cannot skip recovery states to go back to HEALTHY."""
        with pytest.raises(InvalidTransitionError):
            alarmed_machine.transition(DecisionState.HEALTHY)

    def test_stable_to_alarmed_invalid(self, monitor_30m_machine):
        """Cannot go from STABLE back to ALARMED directly."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        monitor_30m_machine.transition(DecisionState.STABLE)
        with pytest.raises(InvalidTransitionError):
            monitor_30m_machine.transition(DecisionState.ALARMED)

    def test_escalated_to_healthy_invalid(self, escalated_machine):
        """Cannot skip recovery from ESCALATED to HEALTHY."""
        with pytest.raises(InvalidTransitionError):
            escalated_machine.transition(DecisionState.HEALTHY)

    def test_escalated_to_monitoring_invalid(self, escalated_machine):
        """Cannot go back to MONITORING from ESCALATED."""
        with pytest.raises(InvalidTransitionError):
            escalated_machine.transition(DecisionState.MONITORING)


# ---------------------------------------------------------------------------
# TestAntiPatternGuard
# ---------------------------------------------------------------------------
class TestAntiPatternGuard:
    """EEMUA 191: Same option max 3 times (CONTROLLED_STOP exempt)."""

    def test_option_first_recommendation(self, alarmed_machine):
        """First recommendation of an option should succeed."""
        result = alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        assert result is not None

    def test_option_second_recommendation(self, alarmed_machine):
        """Second recommendation of same option should succeed."""
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        result = alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        assert result is not None

    def test_option_third_recommendation(self, alarmed_machine):
        """Third recommendation of same option should succeed (boundary)."""
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        result = alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        assert result is not None

    def test_option_fourth_recommendation_rejected(self, alarmed_machine):
        """Fourth recommendation of same option should be rejected (EEMUA 191)."""
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        with pytest.raises(AntiPatternViolation):
            alarmed_machine.recommend_option("SLOWDOWN_ORDER")

    def test_controlled_stop_always_allowed(self, alarmed_machine):
        """CONTROLLED_STOP is exempt from anti-pattern guard."""
        for _ in range(10):
            result = alarmed_machine.recommend_option(CONTROLLED_STOP_OPTION)
            assert result is not None

    def test_different_options_independent_count(self, alarmed_machine):
        """Different options have independent counters."""
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        # Different option should still work
        result = alarmed_machine.recommend_option("STOP_PREP_ORDER")
        assert result is not None

    def test_max_option_repeats_constant(self):
        """MAX_OPTION_REPEATS should be 3 (EEMUA 191)."""
        assert MAX_OPTION_REPEATS == 3

    def test_option_counter_reset_on_state_change(self, alarmed_machine):
        """Option counter should reset when state changes."""
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        # Transition to MONITOR_30M should reset counters
        alarmed_machine.transition(DecisionState.MONITOR_30M)
        # Should be able to recommend again
        result = alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        assert result is not None


# ---------------------------------------------------------------------------
# TestWatchdogTimer
# ---------------------------------------------------------------------------
class TestWatchdogTimer:
    """30-minute watchdog: starts on MONITOR_30M, expiry triggers escalation."""

    def test_watchdog_starts_on_monitor_30m(self, alarmed_machine):
        """Entering MONITOR_30M should start the watchdog timer."""
        alarmed_machine.transition(DecisionState.MONITOR_30M)
        assert alarmed_machine.watchdog_active is True

    def test_watchdog_deadline_set(self, alarmed_machine):
        """Watchdog should set a deadline ~30 minutes in the future."""
        before = datetime.utcnow()
        alarmed_machine.transition(DecisionState.MONITOR_30M)
        deadline = alarmed_machine.watchdog_deadline
        assert deadline is not None
        expected_deadline = before + timedelta(minutes=WATCHDOG_DURATION_MINUTES)
        # Allow 2-second tolerance for test execution time
        assert abs((deadline - expected_deadline).total_seconds()) < 2

    def test_watchdog_duration_constant(self):
        """WATCHDOG_DURATION_MINUTES should be 30."""
        assert WATCHDOG_DURATION_MINUTES == 30

    def test_watchdog_expired_triggers_escalation(self, monitor_30m_machine):
        """After 30 minutes, watchdog should trigger ESCALATED."""
        # Simulate time passing beyond watchdog
        future = datetime.utcnow() + timedelta(minutes=31)
        result = monitor_30m_machine.watchdog_check(now=future)
        assert result == DecisionState.ESCALATED

    def test_watchdog_not_expired_stays(self, monitor_30m_machine):
        """Before 30 minutes, watchdog should not trigger."""
        future = datetime.utcnow() + timedelta(minutes=15)
        result = monitor_30m_machine.watchdog_check(now=future)
        assert result is None or result == DecisionState.MONITOR_30M

    def test_watchdog_stops_on_recovery(self, monitor_30m_machine):
        """Watchdog should stop when transitioning to RECOVERED."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        assert monitor_30m_machine.watchdog_active is False

    def test_watchdog_stops_on_escalation(self, monitor_30m_machine):
        """Watchdog should stop after escalation."""
        monitor_30m_machine.transition(DecisionState.ESCALATED)
        assert monitor_30m_machine.watchdog_active is False

    def test_watchdog_exact_boundary(self, monitor_30m_machine):
        """At exactly 30 minutes, behavior should be deterministic."""
        future = datetime.utcnow() + timedelta(minutes=WATCHDOG_DURATION_MINUTES)
        result = monitor_30m_machine.watchdog_check(now=future)
        # At exact boundary, should trigger escalation
        assert result == DecisionState.ESCALATED


# ---------------------------------------------------------------------------
# TestStableDueAndHealthyDue
# ---------------------------------------------------------------------------
class TestStableDueAndHealthyDue:
    """Time-based checks for STABLE and HEALTHY transitions."""

    def test_stable_due_after_duration(self, monitor_30m_machine):
        """After STABLE_DURATION_MINUTES in RECOVERED, stable_due() = True."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        future = datetime.utcnow() + timedelta(minutes=STABLE_DURATION_MINUTES + 1)
        assert monitor_30m_machine.stable_due(now=future) is True

    def test_stable_not_due_yet(self, monitor_30m_machine):
        """Before STABLE_DURATION_MINUTES, stable_due() = False."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        future = datetime.utcnow() + timedelta(minutes=5)
        assert monitor_30m_machine.stable_due(now=future) is False

    def test_healthy_due_after_stable_duration(self, monitor_30m_machine):
        """After HEALTHY_DURATION_MINUTES in STABLE, healthy_due() = True."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        monitor_30m_machine.transition(DecisionState.STABLE)
        future = datetime.utcnow() + timedelta(minutes=HEALTHY_DURATION_MINUTES + 1)
        assert monitor_30m_machine.healthy_due(now=future) is True

    def test_healthy_not_due_yet(self, monitor_30m_machine):
        """Before HEALTHY_DURATION_MINUTES in STABLE, healthy_due() = False."""
        monitor_30m_machine.transition(DecisionState.RECOVERED)
        monitor_30m_machine.transition(DecisionState.STABLE)
        future = datetime.utcnow() + timedelta(minutes=5)
        assert monitor_30m_machine.healthy_due(now=future) is False

    def test_stable_due_wrong_state(self, machine):
        """stable_due() in wrong state should return False."""
        assert machine.stable_due() is False

    def test_healthy_due_wrong_state(self, machine):
        """healthy_due() in wrong state should return False."""
        assert machine.healthy_due() is False


# ---------------------------------------------------------------------------
# TestResetAlarm
# ---------------------------------------------------------------------------
class TestResetAlarm:
    """reset_alarm() should clean up all alarm state."""

    def test_reset_alarm_from_alarmed(self, alarmed_machine):
        """reset_alarm from ALARMED should return to HEALTHY."""
        alarmed_machine.reset_alarm()
        assert alarmed_machine.state == DecisionState.HEALTHY

    def test_reset_alarm_clears_watchdog(self, monitor_30m_machine):
        """reset_alarm should stop the watchdog."""
        monitor_30m_machine.reset_alarm()
        assert monitor_30m_machine.watchdog_active is False

    def test_reset_alarm_clears_option_history(self, alarmed_machine):
        """reset_alarm should clear option recommendation history."""
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        alarmed_machine.reset_alarm()
        # After reset, should be able to recommend again
        result = alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        assert result is not None

    def test_reset_alarm_from_healthy(self, machine):
        """reset_alarm from HEALTHY should be a no-op (stay HEALTHY)."""
        machine.reset_alarm()
        assert machine.state == DecisionState.HEALTHY

    def test_reset_alarm_from_escalated(self, escalated_machine):
        """reset_alarm from ESCALATED should return to HEALTHY."""
        escalated_machine.reset_alarm()
        assert escalated_machine.state == DecisionState.HEALTHY


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases: unknown machine_id, duplicate transitions, etc."""

    def test_unknown_machine_id(self):
        """Empty machine_id should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            DecisionStateMachine(machine_id="")

    def test_none_machine_id(self):
        """None machine_id should raise TypeError."""
        with pytest.raises((ValueError, TypeError)):
            DecisionStateMachine(machine_id=None)

    def test_duplicate_transition_same_state(self, machine):
        """Transitioning to the same state should be rejected or no-op."""
        machine.transition(DecisionState.MONITORING)
        with pytest.raises(InvalidTransitionError):
            machine.transition(DecisionState.MONITORING)

    def test_initial_state_is_healthy(self, machine):
        """New machine should start in HEALTHY state."""
        assert machine.state == DecisionState.HEALTHY

    def test_state_history_tracked(self, machine):
        """State transitions should be tracked in history."""
        machine.transition(DecisionState.MONITORING)
        machine.transition(DecisionState.ALARMED)
        history = machine.state_history
        assert len(history) >= 2

    def test_concurrent_machine_independence(self):
        """Two machines should have independent state."""
        m1 = DecisionStateMachine(machine_id="AC-001")
        m2 = DecisionStateMachine(machine_id="AC-002")
        m1.transition(DecisionState.MONITORING)
        assert m1.state == DecisionState.MONITORING
        assert m2.state == DecisionState.HEALTHY

    def test_option_record_has_timestamp(self, alarmed_machine):
        """Option recommendation should record timestamp."""
        record = alarmed_machine.recommend_option("SLOWDOWN_ORDER")
        assert hasattr(record, "timestamp")
        assert record.timestamp is not None

    def test_option_record_has_option_name(self, alarmed_machine):
        """Option record should include the option name."""
        record = alarmed_machine.recommend_option("STOP_PREP_ORDER")
        assert record.option_name == "STOP_PREP_ORDER"
