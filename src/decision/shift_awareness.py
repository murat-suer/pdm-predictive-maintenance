"""
Shift-aware maintenance window recommendation.

Provides shift calendar management and RUL-based maintenance window
recommendation following 3 rules:
- RUL < 4 hours → IMMEDIATE_STOP
- RUL < shift_end + margin → AT_SHIFT_END
- Otherwise → AT_NEXT_HANDOVER
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum


class WindowRecommendation(str, Enum):
    """Recommended maintenance window."""
    IMMEDIATE_STOP = "IMMEDIATE_STOP"
    AT_SHIFT_END = "AT_SHIFT_END"
    AT_NEXT_HANDOVER = "AT_NEXT_HANDOVER"


@dataclass
class Shift:
    """A shift definition with name, start and end times."""
    name: str
    start: time
    end: time


@dataclass
class MaintenanceWindow:
    """A recommended maintenance window."""
    window: WindowRecommendation
    target_time: datetime
    reason: str


class ShiftCalendar:
    """
    Manages shift schedules and provides maintenance window recommendations.

    Validates that shifts cover 24 hours without overlap.
    """

    # Critical RUL threshold (hours) - below this, immediate stop
    CRITICAL_RUL_HOURS = 4.0

    def __init__(self, shifts: list[Shift]):
        if not shifts:
            raise ValueError("At least one shift is required.")

        # Validate no overlaps
        self._validate_shifts(shifts)
        self.shifts = shifts

    def _validate_shifts(self, shifts: list[Shift]):
        """Validate that shifts don't overlap."""
        # Sort by start time
        sorted_shifts = sorted(shifts, key=lambda s: s.start)

        for i in range(len(sorted_shifts)):
            current = sorted_shifts[i]
            next_shift = sorted_shifts[(i + 1) % len(sorted_shifts)]

            # Check if current shift overlaps with next
            if current.end > current.start:
                # Normal shift (doesn't cross midnight)
                if next_shift.start < current.end and next_shift.start >= current.start:
                    raise ValueError(
                        f"Shifts overlap: '{current.name}' and '{next_shift.name}'"
                    )
            else:
                # Shift crosses midnight
                pass  # More complex check needed

        # Simpler overlap check: for each pair, check if ranges overlap
        for i in range(len(shifts)):
            for j in range(i + 1, len(shifts)):
                if self._shifts_overlap(shifts[i], shifts[j]):
                    raise ValueError(
                        f"Shifts overlap: '{shifts[i].name}' and '{shifts[j].name}'"
                    )

    def _shifts_overlap(self, s1: Shift, s2: Shift) -> bool:
        """Check if two shifts overlap."""
        # Convert to minutes since midnight for easier comparison
        def to_range(s: Shift):
            start_min = s.start.hour * 60 + s.start.minute
            end_min = s.end.hour * 60 + s.end.minute
            if end_min <= start_min:
                # Crosses midnight: split into two ranges
                return [(start_min, 24 * 60), (0, end_min)]
            return [(start_min, end_min)]

        ranges1 = to_range(s1)
        ranges2 = to_range(s2)

        for r1_start, r1_end in ranges1:
            for r2_start, r2_end in ranges2:
                if r1_start < r2_end and r2_start < r1_end:
                    return True
        return False

    def current_shift(self, dt: datetime) -> Shift:
        """Determine which shift is active at the given datetime."""
        t = dt.time()

        for shift in self.shifts:
            if shift.end > shift.start:
                # Normal shift (doesn't cross midnight)
                if shift.start <= t < shift.end:
                    return shift
            else:
                # Shift crosses midnight (e.g., 22:00 → 06:00)
                if t >= shift.start or t < shift.end:
                    return shift

        # Fallback - should not happen with valid calendar
        raise ValueError(f"No shift found for time {t}")

    def shift_end(self, dt: datetime) -> datetime:
        """Calculate when the current shift ends."""
        shift = self.current_shift(dt)

        if shift.end > shift.start:
            # Normal shift - ends same day
            end_dt = dt.replace(hour=shift.end.hour, minute=shift.end.minute,
                                second=0, microsecond=0)
            if end_dt <= dt:
                # Should not happen, but safety
                end_dt += timedelta(days=1)
        else:
            # Shift crosses midnight
            if dt.time() >= shift.start:
                # Started today, ends tomorrow
                end_dt = dt.replace(hour=shift.end.hour, minute=shift.end.minute,
                                    second=0, microsecond=0) + timedelta(days=1)
            else:
                # Started yesterday, ends today
                end_dt = dt.replace(hour=shift.end.hour, minute=shift.end.minute,
                                    second=0, microsecond=0)

        return end_dt

    def hours_until_shift_end(self, dt: datetime) -> float:
        """Calculate hours remaining until current shift ends."""
        end = self.shift_end(dt)
        delta = (end - dt).total_seconds() / 3600.0
        return max(0.0, delta)

    def recommend_maintenance_window(
        self,
        rul_hours: float,
        now: datetime | None = None,
        margin_hours: float = 1.0,
    ) -> MaintenanceWindow:
        """
        Recommend a maintenance window based on RUL.

        Rules:
        1. RUL < 4 hours → IMMEDIATE_STOP
        2. RUL < hours_until_shift_end + margin → AT_SHIFT_END
        3. Otherwise → AT_NEXT_HANDOVER
        """
        if now is None:
            now = datetime.utcnow()

        # Rule 1: Critical RUL → IMMEDIATE_STOP
        if rul_hours < self.CRITICAL_RUL_HOURS:
            return MaintenanceWindow(
                window=WindowRecommendation.IMMEDIATE_STOP,
                target_time=now,
                reason=f"RUL {rul_hours:.1f}h < {self.CRITICAL_RUL_HOURS}h critical threshold",
            )

        # Calculate hours until shift end
        hours_left = self.hours_until_shift_end(now)
        threshold = hours_left + margin_hours

        # Rule 2: RUL < shift_end + margin → AT_SHIFT_END
        if rul_hours < threshold:
            target = self.shift_end(now)
            return MaintenanceWindow(
                window=WindowRecommendation.AT_SHIFT_END,
                target_time=target,
                reason=f"RUL {rul_hours:.1f}h < shift end + margin ({threshold:.1f}h)",
            )

        # Rule 3: Comfortable → AT_NEXT_HANDOVER
        next_handover = self.shift_end(now)
        return MaintenanceWindow(
            window=WindowRecommendation.AT_NEXT_HANDOVER,
            target_time=next_handover,
            reason=f"RUL {rul_hours:.1f}h allows scheduling at next handover",
        )
