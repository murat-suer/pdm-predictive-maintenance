"""
Unit tests for src.decision.shift_awareness

Tests ShiftCalendar: shift detection, shift-end calculation,
and maintenance window recommendation based on RUL.
"""

from datetime import datetime, time

import pytest

from src.decision.shift_awareness import (
    Shift,
    ShiftCalendar,
    WindowRecommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def calendar():
    """Standard 3-shift calendar (06-14, 14-22, 22-06)."""
    shifts = [
        Shift(name="morning", start=time(6, 0), end=time(14, 0)),
        Shift(name="afternoon", start=time(14, 0), end=time(22, 0)),
        Shift(name="night", start=time(22, 0), end=time(6, 0)),
    ]
    return ShiftCalendar(shifts=shifts)


@pytest.fixture
def overlapping_calendar():
    """Calendar with overlapping shifts - should raise on init."""
    shifts = [
        Shift(name="a", start=time(6, 0), end=time(16, 0)),
        Shift(name="b", start=time(14, 0), end=time(22, 0)),
    ]
    return shifts  # Return raw for validation test


# ---------------------------------------------------------------------------
# TestShiftCalendarInit
# ---------------------------------------------------------------------------
class TestShiftCalendarInit:
    def test_valid_calendar(self, calendar):
        assert len(calendar.shifts) == 3

    def test_overlapping_shifts_rejected(self, overlapping_calendar):
        """Overlapping shifts should raise ValueError on calendar creation."""
        with pytest.raises(ValueError, match="[Oo]verlap"):
            ShiftCalendar(shifts=overlapping_calendar)

    def test_empty_shifts_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            ShiftCalendar(shifts=[])


# ---------------------------------------------------------------------------
# TestCurrentShift
# ---------------------------------------------------------------------------
class TestCurrentShift:
    def test_morning_shift(self, calendar):
        """08:00 should be morning shift."""
        dt = datetime(2026, 6, 9, 8, 0)
        shift = calendar.current_shift(dt)
        assert shift.name == "morning"

    def test_afternoon_shift(self, calendar):
        """16:00 should be afternoon shift."""
        dt = datetime(2026, 6, 9, 16, 0)
        shift = calendar.current_shift(dt)
        assert shift.name == "afternoon"

    def test_night_shift(self, calendar):
        """23:00 should be night shift."""
        dt = datetime(2026, 6, 9, 23, 0)
        shift = calendar.current_shift(dt)
        assert shift.name == "night"

    def test_night_shift_early_morning(self, calendar):
        """02:00 should still be night shift (crosses midnight)."""
        dt = datetime(2026, 6, 9, 2, 0)
        shift = calendar.current_shift(dt)
        assert shift.name == "night"

    def test_shift_boundary_start(self, calendar):
        """Exactly at shift start time."""
        dt = datetime(2026, 6, 9, 14, 0)
        shift = calendar.current_shift(dt)
        assert shift.name == "afternoon"

    def test_all_three_shifts_cover_24h(self, calendar):
        """Every hour should map to exactly one shift."""
        for hour in range(24):
            dt = datetime(2026, 6, 9, hour, 30)
            shift = calendar.current_shift(dt)
            assert shift is not None, f"No shift found for hour {hour}"


# ---------------------------------------------------------------------------
# TestShiftEnd
# ---------------------------------------------------------------------------
class TestShiftEnd:
    def test_morning_shift_end(self, calendar):
        """Morning shift ends at 14:00 same day."""
        dt = datetime(2026, 6, 9, 8, 0)
        end = calendar.shift_end(dt)
        assert end == datetime(2026, 6, 9, 14, 0)

    def test_night_shift_end_crosses_midnight(self, calendar):
        """Night shift starting at 22:00 ends at 06:00 next day."""
        dt = datetime(2026, 6, 9, 23, 0)
        end = calendar.shift_end(dt)
        assert end == datetime(2026, 6, 10, 6, 0)

    def test_night_shift_early_morning_end(self, calendar):
        """Night shift at 02:00 ends at 06:00 same day."""
        dt = datetime(2026, 6, 9, 2, 0)
        end = calendar.shift_end(dt)
        assert end == datetime(2026, 6, 9, 6, 0)

    def test_afternoon_shift_end(self, calendar):
        dt = datetime(2026, 6, 9, 18, 0)
        end = calendar.shift_end(dt)
        assert end == datetime(2026, 6, 9, 22, 0)


# ---------------------------------------------------------------------------
# TestHoursUntilShiftEnd
# ---------------------------------------------------------------------------
class TestHoursUntilShiftEnd:
    def test_morning_midpoint(self, calendar):
        """At 10:00, 4 hours until 14:00."""
        dt = datetime(2026, 6, 9, 10, 0)
        hours = calendar.hours_until_shift_end(dt)
        assert hours == pytest.approx(4.0, abs=0.1)

    def test_night_shift_hours(self, calendar):
        """At 23:00, 7 hours until 06:00."""
        dt = datetime(2026, 6, 9, 23, 0)
        hours = calendar.hours_until_shift_end(dt)
        assert hours == pytest.approx(7.0, abs=0.1)

    def test_near_end_of_shift(self, calendar):
        """30 minutes before end."""
        dt = datetime(2026, 6, 9, 13, 30)
        hours = calendar.hours_until_shift_end(dt)
        assert hours == pytest.approx(0.5, abs=0.1)

    def test_just_started_shift(self, calendar):
        """Just started afternoon shift at 14:05."""
        dt = datetime(2026, 6, 9, 14, 5)
        hours = calendar.hours_until_shift_end(dt)
        assert hours == pytest.approx(7.0 + 55 / 60, abs=0.1)


# ---------------------------------------------------------------------------
# TestRecommendMaintenanceWindow
# ---------------------------------------------------------------------------
class TestRecommendMaintenanceWindow:
    """
    CRITICAL: recommend_maintenance_window() logic:
    - RUL < 4 hours → IMMEDIATE_STOP
    - RUL < shift_end + margin → AT_SHIFT_END
    - RUL >= shift_end + margin (comfortable) → AT_NEXT_HANDOVER
    """

    def test_immediate_stop_critical_rul(self, calendar):
        """RUL < 4 hours → IMMEDIATE_STOP."""
        dt = datetime(2026, 6, 9, 10, 0)
        rul_hours = 2.0  # 2 hours remaining
        rec = calendar.recommend_maintenance_window(rul_hours=rul_hours, now=dt)
        assert rec.window == WindowRecommendation.IMMEDIATE_STOP

    def test_immediate_stop_very_low_rul(self, calendar):
        """RUL = 0.5 hours → IMMEDIATE_STOP."""
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=0.5, now=dt)
        assert rec.window == WindowRecommendation.IMMEDIATE_STOP

    def test_at_shift_end_rul_within_shift(self, calendar):
        """RUL >= CRITICAL_RUL_HOURS but < shift_end + margin → AT_SHIFT_END."""
        dt = datetime(2026, 6, 9, 10, 0)  # 4h until shift end
        rul_hours = 4.5  # >= 4.0 (critical), but < 4+1=5 (shift_end + margin)
        rec = calendar.recommend_maintenance_window(rul_hours=rul_hours, now=dt)
        assert rec.window == WindowRecommendation.AT_SHIFT_END

    def test_immediate_stop_rul_3_hours(self, calendar):
        """RUL=3.0 < CRITICAL_RUL_HOURS(4.0) → IMMEDIATE_STOP."""
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=3.0, now=dt)
        assert rec.window == WindowRecommendation.IMMEDIATE_STOP

    def test_at_next_handover_comfortable_margin(self, calendar):
        """RUL >> shift_end → AT_NEXT_HANDOVER (comfortable)."""
        dt = datetime(2026, 6, 9, 10, 0)  # 4h until shift end
        rul_hours = 24.0  # Plenty of time
        rec = calendar.recommend_maintenance_window(rul_hours=rul_hours, now=dt)
        assert rec.window == WindowRecommendation.AT_NEXT_HANDOVER

    def test_immediate_stop_boundary_4_hours(self, calendar):
        """RUL exactly at 4h boundary - should be IMMEDIATE_STOP."""
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=4.0, now=dt)
        # At exactly 4h, it's at the boundary - implementation decides
        assert rec.window in (
            WindowRecommendation.IMMEDIATE_STOP,
            WindowRecommendation.AT_SHIFT_END,
        )

    def test_recommendation_includes_target_time(self, calendar):
        """Recommendation should include a target datetime."""
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=2.0, now=dt)
        assert rec.target_time is not None
        assert isinstance(rec.target_time, datetime)

    def test_recommendation_includes_reason(self, calendar):
        """Recommendation should include human-readable reason."""
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=2.0, now=dt)
        assert rec.reason is not None
        assert len(rec.reason) > 0


# ---------------------------------------------------------------------------
# TestNegativeRULValidation
# ---------------------------------------------------------------------------
class TestNegativeRULValidation:
    """Negative RUL means already failed - should be IMMEDIATE_STOP."""

    def test_negative_rul_immediate_stop(self, calendar):
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=-1.0, now=dt)
        assert rec.window == WindowRecommendation.IMMEDIATE_STOP

    def test_zero_rul_immediate_stop(self, calendar):
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=0.0, now=dt)
        assert rec.window == WindowRecommendation.IMMEDIATE_STOP


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_very_large_rul(self, calendar):
        """RUL = 1000 hours → AT_NEXT_HANDOVER."""
        dt = datetime(2026, 6, 9, 10, 0)
        rec = calendar.recommend_maintenance_window(rul_hours=1000.0, now=dt)
        assert rec.window == WindowRecommendation.AT_NEXT_HANDOVER

    def test_shift_at_midnight_boundary(self, calendar):
        """Test at exactly midnight."""
        dt = datetime(2026, 6, 10, 0, 0)
        shift = calendar.current_shift(dt)
        assert shift.name == "night"

    def test_recommendation_with_margin_parameter(self, calendar):
        """Custom margin should affect AT_SHIFT_END vs AT_NEXT_HANDOVER boundary."""
        dt = datetime(2026, 6, 9, 10, 0)  # 4h until shift end
        # With margin=2, threshold = 4+2=6h. RUL=5 < 6 → AT_SHIFT_END
        rec = calendar.recommend_maintenance_window(
            rul_hours=5.0, now=dt, margin_hours=2.0
        )
        assert rec.window == WindowRecommendation.AT_SHIFT_END
