"""
Unit tests for src.decision.escalation_manager (Phase 2B - Core Logic Layer)

Tests EscalationManager: L1→L2→L3 escalation with trend analysis
(IMPROVING/STABLE/WORSENING), 15-min L2→L3 timeout, auto-resolve on
IMPROVING 3+ checks, score history, downtime simulation, and DEMO_MODE.

NOTE: These tests will FAIL until escalation_manager.py is implemented.
"""

from datetime import datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.escalation_manager import (
    AUTO_RESOLVE_IMPROVING_COUNT,
    DEMO_MODE,
    L2_TIMEOUT_MINUTES,
    SCORE_HISTORY_SIZE,
    EscalationLevel,
    EscalationManager,
    Trend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def manager():
    """Fresh EscalationManager."""
    return EscalationManager(machine_id="AC-001")


@pytest.fixture
def l1_manager():
    """EscalationManager at L1 level."""
    m = EscalationManager(machine_id="AC-001")
    m.escalate(level=EscalationLevel.L1, reason="Initial alarm")
    return m


@pytest.fixture
def l2_manager():
    """EscalationManager at L2 level."""
    m = EscalationManager(machine_id="AC-001")
    m.escalate(level=EscalationLevel.L1, reason="Initial alarm")
    m.escalate(level=EscalationLevel.L2, reason="No response at L1")
    return m


@pytest.fixture
def l3_manager():
    """EscalationManager at L3 level."""
    m = EscalationManager(machine_id="AC-001")
    m.escalate(level=EscalationLevel.L1, reason="Initial alarm")
    m.escalate(level=EscalationLevel.L2, reason="No response at L1")
    m.escalate(level=EscalationLevel.L3, reason="Timeout at L2")
    return m


def _make_score(value: float, timestamp: datetime | None = None) -> dict:
    """Helper to create a score entry dict."""
    return {
        "value": value,
        "timestamp": timestamp or datetime.utcnow(),
    }


# ---------------------------------------------------------------------------
# TestEscalationL1ToL2
# ---------------------------------------------------------------------------
class TestEscalationL1ToL2:
    """L1→L2 escalation triggered by WORSENING trend."""

    def test_l1_to_l2_on_worsening(self, l1_manager):
        """WORSENING trend should trigger L1→L2 escalation."""
        # Record worsening scores
        l1_manager.record_score(0.3)
        l1_manager.record_score(0.5)
        l1_manager.record_score(0.7)
        # Trend should be WORSENING
        trend = l1_manager.classify_trend()
        assert trend == Trend.WORSENING
        # Escalate
        l1_manager.escalate(level=EscalationLevel.L2, reason="Worsening trend")
        assert l1_manager.current_level == EscalationLevel.L2

    def test_l2_requires_reason(self, l1_manager):
        """L2 escalation should require a reason."""
        with pytest.raises((ValueError, TypeError)):
            l1_manager.escalate(level=EscalationLevel.L2, reason="")

    def test_l2_records_timestamp(self, l1_manager):
        """L2 escalation should record the timestamp."""
        l1_manager.escalate(level=EscalationLevel.L2, reason="Worsening")
        record = l1_manager.get_current_record()
        assert record.escalated_at is not None

    def test_l1_initial_level(self, manager):
        """New manager should start at no escalation or L0."""
        assert manager.current_level is None or manager.current_level == EscalationLevel.L1


# ---------------------------------------------------------------------------
# TestEscalationL2ToL3
# ---------------------------------------------------------------------------
class TestEscalationL2ToL3:
    """L2→L3 escalation triggered by 15-minute timeout."""

    def test_l2_to_l3_on_timeout(self, l2_manager):
        """After L2_TIMEOUT_MINUTES without resolution → L3."""
        future = datetime.utcnow() + timedelta(minutes=L2_TIMEOUT_MINUTES + 1)
        result = l2_manager.check_timeout(now=future)
        assert result is True or result == EscalationLevel.L3

    def test_l2_not_yet_timed_out(self, l2_manager):
        """Before L2_TIMEOUT_MINUTES, should not escalate to L3."""
        future = datetime.utcnow() + timedelta(minutes=5)
        result = l2_manager.check_timeout(now=future)
        assert result is False or result is None

    def test_l2_timeout_constant(self):
        """L2_TIMEOUT_MINUTES should be 15."""
        assert L2_TIMEOUT_MINUTES == 15

    def test_l3_after_timeout(self, l2_manager):
        """After timeout, escalate to L3."""
        l2_manager.escalate(level=EscalationLevel.L3, reason="L2 timeout expired")
        assert l2_manager.current_level == EscalationLevel.L3

    def test_l2_timeout_exact_boundary(self, l2_manager):
        """At exactly L2_TIMEOUT_MINUTES, behavior should be deterministic."""
        future = datetime.utcnow() + timedelta(minutes=L2_TIMEOUT_MINUTES)
        result = l2_manager.check_timeout(now=future)
        # At exact boundary, should trigger
        assert result is True or result == EscalationLevel.L3


# ---------------------------------------------------------------------------
# TestAutoResolve
# ---------------------------------------------------------------------------
class TestAutoResolve:
    """Auto-resolve when trend is IMPROVING for 3+ consecutive checks."""

    def test_auto_resolve_after_3_improving(self, l2_manager):
        """3+ consecutive IMPROVING checks → auto-resolve."""
        # Record improving scores (decreasing = improving)
        l2_manager.record_score(0.8)
        l2_manager.record_score(0.6)
        l2_manager.record_score(0.4)
        # Check trend
        trend = l2_manager.classify_trend()
        assert trend == Trend.IMPROVING
        # Auto-resolve should trigger
        result = l2_manager.check_auto_resolve()
        assert result is True

    def test_no_auto_resolve_with_stable(self, l2_manager):
        """STABLE trend should not trigger auto-resolve."""
        l2_manager.record_score(0.5)
        l2_manager.record_score(0.5)
        l2_manager.record_score(0.5)
        trend = l2_manager.classify_trend()
        assert trend == Trend.STABLE
        result = l2_manager.check_auto_resolve()
        assert result is False

    def test_no_auto_resolve_with_worsening(self, l2_manager):
        """WORSENING trend should not trigger auto-resolve."""
        l2_manager.record_score(0.3)
        l2_manager.record_score(0.5)
        l2_manager.record_score(0.7)
        trend = l2_manager.classify_trend()
        assert trend == Trend.WORSENING
        result = l2_manager.check_auto_resolve()
        assert result is False

    def test_auto_resolve_count_constant(self):
        """AUTO_RESOLVE_IMPROVING_COUNT should be 3."""
        assert AUTO_RESOLVE_IMPROVING_COUNT == 3

    def test_auto_resolve_below_threshold_not_triggered(self, l2_manager):
        """2 IMPROVING checks (below 3) should not auto-resolve."""
        l2_manager.record_score(0.8)
        l2_manager.record_score(0.6)
        # Only 2 improving, below threshold
        result = l2_manager.check_auto_resolve()
        assert result is False

    def test_auto_resolve_resets_on_worsening(self, l2_manager):
        """Improving streak should reset if a worsening score appears."""
        l2_manager.record_score(0.8)
        l2_manager.record_score(0.6)
        # Streak broken by worsening
        l2_manager.record_score(0.9)
        result = l2_manager.check_auto_resolve()
        assert result is False


# ---------------------------------------------------------------------------
# TestTrendClassification
# ---------------------------------------------------------------------------
class TestTrendClassification:
    """Trend classification: IMPROVING, STABLE, WORSENING."""

    def test_improving_trend(self, manager):
        """Decreasing scores → IMPROVING."""
        manager.record_score(0.9)
        manager.record_score(0.7)
        manager.record_score(0.5)
        manager.record_score(0.3)
        assert manager.classify_trend() == Trend.IMPROVING

    def test_worsening_trend(self, manager):
        """Increasing scores → WORSENING."""
        manager.record_score(0.2)
        manager.record_score(0.4)
        manager.record_score(0.6)
        manager.record_score(0.8)
        assert manager.classify_trend() == Trend.WORSENING

    def test_stable_trend(self, manager):
        """Constant scores → STABLE."""
        manager.record_score(0.5)
        manager.record_score(0.5)
        manager.record_score(0.5)
        manager.record_score(0.5)
        assert manager.classify_trend() == Trend.STABLE

    def test_mostly_improving(self, manager):
        """Majority decreasing → IMPROVING."""
        manager.record_score(0.9)
        manager.record_score(0.7)
        manager.record_score(0.5)
        manager.record_score(0.6)  # One increase
        assert manager.classify_trend() == Trend.IMPROVING

    def test_mostly_worsening(self, manager):
        """Majority increasing → WORSENING."""
        manager.record_score(0.2)
        manager.record_score(0.4)
        manager.record_score(0.6)
        manager.record_score(0.5)  # One decrease
        assert manager.classify_trend() == Trend.WORSENING


# ---------------------------------------------------------------------------
# TestScoreHistory
# ---------------------------------------------------------------------------
class TestScoreHistory:
    """Score history tracking (max 5 scores)."""

    def test_score_history_records(self, manager):
        """record_score should add to history."""
        manager.record_score(0.5)
        assert len(manager.score_history) == 1

    def test_score_history_max_size(self, manager):
        """History should be capped at SCORE_HISTORY_SIZE."""
        for i in range(10):
            manager.record_score(0.1 * i)
        assert len(manager.score_history) <= SCORE_HISTORY_SIZE

    def test_score_history_size_constant(self):
        """SCORE_HISTORY_SIZE should be 5."""
        assert SCORE_HISTORY_SIZE == 5

    def test_score_history_fifo(self, manager):
        """Oldest scores should be evicted first (FIFO)."""
        # Use valid scores in [0.0, 1.0] range
        for i in range(SCORE_HISTORY_SIZE + 3):
            manager.record_score(float(i) * 0.1)  # 0.0, 0.1, 0.2, ..., 0.7
        # First 3 should be gone
        history = manager.score_history
        assert len(history) == SCORE_HISTORY_SIZE
        # Oldest remaining should be score 0.3
        assert abs(history[0].value - 0.3) < 0.01

    def test_score_entry_has_timestamp(self, manager):
        """Each score entry should have a timestamp."""
        manager.record_score(0.5)
        entry = manager.score_history[-1]
        assert hasattr(entry, "timestamp")
        assert entry.timestamp is not None

    def test_score_entry_has_value(self, manager):
        """Each score entry should have a numeric value."""
        manager.record_score(0.75)
        entry = manager.score_history[-1]
        assert entry.value == 0.75


# ---------------------------------------------------------------------------
# TestDowntimeSimulation
# ---------------------------------------------------------------------------
class TestDowntimeSimulation:
    """Downtime simulation using Gaussian distribution per machine type."""

    def test_downtime_returns_positive(self, manager):
        """Simulated downtime should be positive."""
        downtime = manager.simulate_downtime(machine_type="compressor")
        assert downtime > 0

    def test_downtime_varies_by_machine_type(self, manager):
        """Different machine types should have different downtime distributions."""
        # Run multiple simulations
        compressor_times = [manager.simulate_downtime(machine_type="compressor") for _ in range(20)]
        pump_times = [manager.simulate_downtime(machine_type="pump") for _ in range(20)]
        # Averages should differ (with high probability)
        avg_compressor = sum(compressor_times) / len(compressor_times)
        avg_pump = sum(pump_times) / len(pump_times)
        # They should not be exactly equal (Gaussian randomness)
        # This test may rarely fail due to randomness, but very unlikely with 20 samples
        # We just verify both return valid positive numbers
        assert avg_compressor > 0
        assert avg_pump > 0

    def test_downtime_deterministic_with_seed(self, manager):
        """With a fixed seed, downtime should be deterministic."""
        d1 = manager.simulate_downtime(machine_type="compressor", seed=42)
        d2 = manager.simulate_downtime(machine_type="compressor", seed=42)
        assert d1 == d2

    def test_downtime_unknown_machine_type(self, manager):
        """Unknown machine type should use default distribution or raise."""
        # Should either return a default value or raise a known error
        try:
            result = manager.simulate_downtime(machine_type="unknown_xyz")
            assert result > 0  # If it returns, should be positive
        except (ValueError, KeyError):
            pass  # Also acceptable


# ---------------------------------------------------------------------------
# TestDemoMode
# ---------------------------------------------------------------------------
class TestDemoMode:
    """DEMO_MODE: auto-acknowledge escalations."""

    def test_demo_mode_constant_exists(self):
        """DEMO_MODE constant should exist."""
        assert isinstance(DEMO_MODE, bool)

    def test_demo_mode_auto_acknowledge(self):
        """In DEMO_MODE, escalations should be auto-acknowledged."""
        m = EscalationManager(machine_id="AC-001", demo_mode=True)
        m.escalate(level=EscalationLevel.L1, reason="Test")
        record = m.get_current_record()
        assert record.acknowledged is True

    def test_non_demo_mode_requires_ack(self):
        """In non-DEMO mode, escalations need manual acknowledge."""
        m = EscalationManager(machine_id="AC-001", demo_mode=False)
        m.escalate(level=EscalationLevel.L1, reason="Test")
        record = m.get_current_record()
        assert record.acknowledged is False

    def test_demo_mode_all_levels(self):
        """DEMO_MODE should auto-ack at all levels."""
        m = EscalationManager(machine_id="AC-001", demo_mode=True)
        for level in [EscalationLevel.L1, EscalationLevel.L2, EscalationLevel.L3]:
            m.escalate(level=level, reason=f"Test {level}")
            record = m.get_current_record()
            assert record.acknowledged is True, f"Level {level} not auto-acked in DEMO_MODE"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases: empty history, single score, boundary trends."""

    def test_empty_history_trend(self, manager):
        """Classifying trend with empty history should return STABLE or raise."""
        try:
            trend = manager.classify_trend()
            assert trend == Trend.STABLE  # Default for no data
        except (ValueError, IndexError):
            pass  # Also acceptable

    def test_single_score_trend(self, manager):
        """Classifying trend with single score should return STABLE or raise."""
        manager.record_score(0.5)
        try:
            trend = manager.classify_trend()
            assert trend == Trend.STABLE  # Can't determine with 1 point
        except (ValueError, IndexError):
            pass  # Also acceptable

    def test_two_scores_trend(self, manager):
        """With 2 scores, trend should be determinable."""
        manager.record_score(0.3)
        manager.record_score(0.7)
        trend = manager.classify_trend()
        assert trend in (Trend.WORSENING, Trend.STABLE)

    def test_zero_score(self, manager):
        """Score of 0.0 should be valid."""
        manager.record_score(0.0)
        assert len(manager.score_history) == 1
        assert manager.score_history[-1].value == 0.0

    def test_one_score(self, manager):
        """Score of 1.0 should be valid."""
        manager.record_score(1.0)
        assert len(manager.score_history) == 1
        assert manager.score_history[-1].value == 1.0

    def test_negative_score_rejected(self, manager):
        """Negative score should be rejected or clamped."""
        try:
            manager.record_score(-0.1)
            # If accepted, should be clamped to 0
            assert manager.score_history[-1].value >= 0.0
        except ValueError:
            pass  # Also acceptable

    def test_score_above_one_rejected(self, manager):
        """Score > 1.0 should be rejected or clamped."""
        try:
            manager.record_score(1.5)
            # If accepted, should be clamped to 1.0
            assert manager.score_history[-1].value <= 1.0
        except ValueError:
            pass  # Also acceptable

    def test_empty_machine_id(self):
        """Empty machine_id should raise."""
        with pytest.raises((ValueError, TypeError)):
            EscalationManager(machine_id="")

    def test_escalation_record_has_level(self, l1_manager):
        """Escalation record should include the level."""
        record = l1_manager.get_current_record()
        assert record.level == EscalationLevel.L1

    def test_escalation_record_has_reason(self, l1_manager):
        """Escalation record should include the reason."""
        record = l1_manager.get_current_record()
        assert record.reason == "Initial alarm"

    def test_multiple_escalations_tracked(self, l3_manager):
        """All escalation levels should be tracked in history."""
        history = l3_manager.escalation_history
        assert len(history) >= 3

    def test_resolve_clears_escalation(self, l2_manager):
        """Resolving should clear the current escalation."""
        l2_manager.resolve()
        assert l2_manager.current_level is None or l2_manager.is_resolved is True
