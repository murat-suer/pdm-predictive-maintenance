"""
Tests for SQLAlchemy event listeners in models.py.

Tests _ensure_decision_log_on_alarm:
- When an AlarmState with status='UNACKNOWLEDGED' is inserted,
  a DecisionLog should be auto-created.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestEnsureDecisionLogOnAlarm:
    def test_ensure_decision_log_on_alarm(self):
        """
        Test that _ensure_decision_log_on_alarm creates a DecisionLog
        when an AlarmState with status='UNACKNOWLEDGED' is inserted.
        """
        from src.database.models import _ensure_decision_log_on_alarm

        # Create a mock alarm target with UNACKNOWLEDGED status
        mock_target = MagicMock()
        mock_target.status = "UNACKNOWLEDGED"
        mock_target.id = 42
        mock_target.machine_id = "CM-203"
        mock_target.level = 2

        # Mock the connection — simulate no existing DecisionLog
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_result.first.return_value = None  # No existing decision log
        mock_connection.execute.return_value = mock_result

        # Mock the mapper (not used in the function)
        mock_mapper = MagicMock()

        # Call the listener
        _ensure_decision_log_on_alarm(mock_mapper, mock_connection, mock_target)

        # Verify that execute was called (to check for existing + to insert)
        assert mock_connection.execute.call_count == 2, (
            "Should call execute twice: once for SELECT check, once for INSERT"
        )

    def test_ensure_decision_log_skips_non_unacknowledged(self):
        """
        Test that _ensure_decision_log_on_alarm does NOT create a DecisionLog
        when the alarm status is not 'UNACKNOWLEDGED'.
        """
        from src.database.models import _ensure_decision_log_on_alarm

        mock_target = MagicMock()
        mock_target.status = "ACKNOWLEDGED"  # Not UNACKNOWLEDGED

        mock_connection = MagicMock()
        mock_mapper = MagicMock()

        _ensure_decision_log_on_alarm(mock_mapper, mock_connection, mock_target)

        # Should NOT call execute at all — early return
        mock_connection.execute.assert_not_called()

    def test_ensure_decision_log_skips_if_exists(self):
        """
        Test that _ensure_decision_log_on_alarm does NOT create a duplicate
        DecisionLog when one already exists for the alarm.
        """
        from src.database.models import _ensure_decision_log_on_alarm

        mock_target = MagicMock()
        mock_target.status = "UNACKNOWLEDGED"
        mock_target.id = 42
        mock_target.machine_id = "CM-203"
        mock_target.level = 1

        mock_connection = MagicMock()
        mock_result = MagicMock()
        # Simulate an existing DecisionLog
        mock_result.first.return_value = MagicMock()
        mock_connection.execute.return_value = mock_result

        mock_mapper = MagicMock()

        _ensure_decision_log_on_alarm(mock_mapper, mock_connection, mock_target)

        # Should call execute only once (the SELECT check), not twice (no INSERT)
        assert mock_connection.execute.call_count == 1, (
            "Should call execute only once (SELECT check) when DecisionLog already exists"
        )
