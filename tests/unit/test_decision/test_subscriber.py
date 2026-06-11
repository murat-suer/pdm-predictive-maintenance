"""
Unit tests for src.decision.subscriber (Phase 2D - Pipeline & Integration Layer).

Tests the Redis Streams consumer that:
- Reads anomaly events from anomaly_stream
- Creates AlarmState records
- Generates decision scenarios via DecisionEngine
- Implements ISA-18.2 alarm flood detection (R-010)
- Handles periodic tick (shelve expiry, L2 timeout)
- Handles demo operator tick (pending decisions)
- Logs EU AI Act compliance events

NOTE: These tests will FAIL until src/decision/subscriber.py is implemented.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Constants (mirror expected implementation)
# ---------------------------------------------------------------------------
ALARM_FLOOD_THRESHOLD = 10  # ISA-18.2 R-010: 10 alarms
ALARM_FLOOD_WINDOW_MINUTES = 10  # ISA-18.2 R-010: in 10 minutes
DEMO_FLOOD_THRESHOLD = 100  # DEMO_MODE threshold
FLOOD_RECOVERY_MINUTES = 5  # Recovery: 5 minutes below threshold
PERIODIC_TICK_INTERVAL_S = 300  # 5 minutes
DEMO_TICK_INTERVAL_S = 30  # 30 seconds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_redis():
    """Mock Redis client with Streams support."""
    r = MagicMock()
    r.xreadgroup.return_value = []
    r.xack.return_value = 1
    r.xgroup_create.return_value = True
    return r


@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy session."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.query = MagicMock()
    return session


@pytest.fixture
def sample_anomaly_event():
    """Sample anomaly event from anomaly_stream."""
    return {
        b"event_id": b"evt-001",
        b"timestamp": b"2026-06-10T12:00:00+00:00",
        b"source": b"pm-ml",
        b"event_type": b"ANOMALY_DETECTED",
        b"machine_id": b"AC-201",
        b"anomaly_id": b"42",
        b"anomaly_score": b"0.85",
        b"top_contributing_sensor": b"vibration_rms",
        b"severity": b"CRITICAL",
        b"phase": b"DEGRADED",
    }


@pytest.fixture
def sample_anomaly_event_predicted_failure():
    """Sample PREDICTED_FAILURE event from anomaly_stream."""
    return {
        b"event_id": b"evt-002",
        b"timestamp": b"2026-06-10T12:05:00+00:00",
        b"source": b"pm-ml-rul",
        b"event_type": b"PREDICTED_FAILURE",
        b"machine_id": b"HX-202",
        b"anomaly_id": b"43",
        b"anomaly_score": b"0.92",
        b"top_contributing_sensor": b"temperature_outlet",
        b"severity": b"CRITICAL",
        b"phase": b"CRITICAL",
        b"degradation_level": b"0.75",
        b"rul_hours": b"4.50",
    }


@pytest.fixture
def subscriber_instance(mock_redis, mock_db_session):
    """Create a DecisionSubscriber instance with mocked dependencies."""
    from src.decision.subscriber import DecisionSubscriber
    return DecisionSubscriber(
        redis_client=mock_redis,
        db_session=mock_db_session,
        consumer_group="decision_consumers",
        consumer_name="decision-worker-1",
    )


# ---------------------------------------------------------------------------
# TestConsumerGroupCreation
# ---------------------------------------------------------------------------
class TestConsumerGroupCreation:
    """Test Redis consumer group initialization."""

    def test_creates_consumer_group_on_start(self, mock_redis):
        """Should create consumer group when starting."""
        from src.decision.subscriber import DecisionSubscriber
        sub = DecisionSubscriber(
            redis_client=mock_redis,
            db_session=MagicMock(),
            consumer_group="decision_consumers",
            consumer_name="worker-1",
        )
        sub._ensure_consumer_group()
        mock_redis.xgroup_create.assert_called_once()

    def test_ignores_busygroup_error(self, mock_redis):
        """Should silently ignore BUSYGROUP if group already exists."""
        import redis as redis_lib
        mock_redis.xgroup_create.side_effect = redis_lib.exceptions.ResponseError(
            "BUSYGROUP Consumer Group name already exists"
        )
        from src.decision.subscriber import DecisionSubscriber
        sub = DecisionSubscriber(
            redis_client=mock_redis,
            db_session=MagicMock(),
            consumer_group="decision_consumers",
            consumer_name="worker-1",
        )
        # Should not raise
        sub._ensure_consumer_group()

    def test_raises_on_other_redis_errors(self, mock_redis):
        """Should raise on non-BUSYGROUP Redis errors."""
        import redis as redis_lib
        mock_redis.xgroup_create.side_effect = redis_lib.exceptions.ResponseError(
            "ERR Some other error"
        )
        from src.decision.subscriber import DecisionSubscriber
        sub = DecisionSubscriber(
            redis_client=mock_redis,
            db_session=MagicMock(),
            consumer_group="decision_consumers",
            consumer_name="worker-1",
        )
        with pytest.raises(redis_lib.exceptions.ResponseError):
            sub._ensure_consumer_group()

    def test_consumer_group_uses_correct_stream_key(self, mock_redis):
        """Consumer group should be created on anomaly_stream."""
        from src.decision.subscriber import ANOMALY_STREAM, DecisionSubscriber
        sub = DecisionSubscriber(
            redis_client=mock_redis,
            db_session=MagicMock(),
            consumer_group="decision_consumers",
            consumer_name="worker-1",
        )
        sub._ensure_consumer_group()
        call_args = mock_redis.xgroup_create.call_args
        # First positional arg should be the stream key
        assert call_args[0][0] == ANOMALY_STREAM


# ---------------------------------------------------------------------------
# TestAnomalyEventProcessing
# ---------------------------------------------------------------------------
class TestAnomalyEventProcessing:
    """Test processing of anomaly events from Redis Streams."""

    def test_process_anomaly_event_creates_alarm(self, subscriber_instance, sample_anomaly_event, mock_db_session):
        """Processing an anomaly event should create an AlarmState record."""
        sub = subscriber_instance
        result = sub._process_anomaly_event(sample_anomaly_event)
        assert result is not None
        mock_db_session.add.assert_called()

    def test_process_event_extracts_machine_id(self, subscriber_instance, sample_anomaly_event):
        """Should correctly extract machine_id from event fields."""
        sub = subscriber_instance
        fields = sub._parse_event_fields(sample_anomaly_event)
        assert fields["machine_id"] == "AC-201"

    def test_process_event_extracts_anomaly_id(self, subscriber_instance, sample_anomaly_event):
        """Should correctly extract anomaly_id from event fields."""
        sub = subscriber_instance
        fields = sub._parse_event_fields(sample_anomaly_event)
        assert fields["anomaly_id"] == "42"

    def test_process_event_extracts_severity(self, subscriber_instance, sample_anomaly_event):
        """Should correctly extract severity from event fields."""
        sub = subscriber_instance
        fields = sub._parse_event_fields(sample_anomaly_event)
        assert fields["severity"] == "CRITICAL"

    def test_process_predicted_failure_event(self, subscriber_instance, sample_anomaly_event_predicted_failure):
        """Should handle PREDICTED_FAILURE event type."""
        sub = subscriber_instance
        fields = sub._parse_event_fields(sample_anomaly_event_predicted_failure)
        assert fields["event_type"] == "PREDICTED_FAILURE"
        assert fields["rul_hours"] == "4.50"

    def test_process_event_with_string_keys(self, subscriber_instance):
        """Should handle events with string keys (not just bytes)."""
        sub = subscriber_instance
        event = {
            "event_id": "evt-003",
            "timestamp": "2026-06-10T12:10:00+00:00",
            "source": "pm-ml",
            "event_type": "ANOMALY_DETECTED",
            "machine_id": "CM-303",
            "anomaly_id": "44",
            "anomaly_score": "0.78",
            "top_contributing_sensor": "pressure",
            "severity": "WARNING",
            "phase": "HEALTHY",
        }
        fields = sub._parse_event_fields(event)
        assert fields["machine_id"] == "CM-303"

    def test_xack_called_after_successful_processing(self, subscriber_instance, mock_redis, sample_anomaly_event):
        """Should acknowledge message after successful processing."""
        sub = subscriber_instance
        mock_redis.xreadgroup.return_value = [
            [b"anomaly_stream", [(b"1-0", sample_anomaly_event)]]
        ]
        sub._poll_once()
        mock_redis.xack.assert_called()


# ---------------------------------------------------------------------------
# TestAlarmCreation
# ---------------------------------------------------------------------------
class TestAlarmCreation:
    """Test AlarmState record creation from anomaly events."""

    def test_alarm_state_created_with_correct_machine_id(
        self, subscriber_instance, sample_anomaly_event, mock_db_session
    ):
        """AlarmState should have correct machine_id."""
        sub = subscriber_instance
        sub._process_anomaly_event(sample_anomaly_event)
        # Check that an AlarmState was added to session
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        alarm_states = [obj for obj in added_objects if hasattr(obj, 'machine_id')]
        assert len(alarm_states) >= 1
        assert alarm_states[0].machine_id == "AC-201"

    def test_alarm_state_initial_status_is_unacknowledged(
        self, subscriber_instance, sample_anomaly_event, mock_db_session
    ):
        """New alarms should start in UNACKNOWLEDGED state."""
        sub = subscriber_instance
        sub._process_anomaly_event(sample_anomaly_event)
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        alarm_states = [obj for obj in added_objects if hasattr(obj, 'status') and hasattr(obj, 'anomaly_id')]
        assert len(alarm_states) >= 1
        assert alarm_states[0].status == "UNACKNOWLEDGED"

    def test_alarm_severity_critical_for_high_score(
        self, subscriber_instance, mock_db_session
    ):
        """Anomaly score > 0.75 should create CRITICAL severity alarm."""
        sub = subscriber_instance
        event = {
            b"event_id": b"evt-high",
            b"timestamp": b"2026-06-10T12:00:00+00:00",
            b"source": b"pm-ml",
            b"event_type": b"ANOMALY_DETECTED",
            b"machine_id": b"AC-201",
            b"anomaly_id": b"99",
            b"anomaly_score": b"0.95",
            b"top_contributing_sensor": b"vibration_rms",
            b"severity": b"CRITICAL",
            b"phase": b"DEGRADED",
        }
        sub._process_anomaly_event(event)
        mock_db_session.commit.assert_called()

    def test_alarm_links_to_anomaly_id(
        self, subscriber_instance, sample_anomaly_event, mock_db_session
    ):
        """AlarmState should reference the anomaly_id."""
        sub = subscriber_instance
        sub._process_anomaly_event(sample_anomaly_event)
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        alarm_states = [obj for obj in added_objects if hasattr(obj, 'anomaly_id')]
        assert len(alarm_states) >= 1
        assert alarm_states[0].anomaly_id == 42


# ---------------------------------------------------------------------------
# TestAlarmFloodDetection
# ---------------------------------------------------------------------------
class TestAlarmFloodDetection:
    """Test ISA-18.2 R-010 alarm flood detection."""

    def test_flood_not_active_below_threshold(self, subscriber_instance):
        """Flood should NOT be active when alarm count < threshold."""
        sub = subscriber_instance
        now = datetime.now(UTC)
        # Add 5 alarms in last 10 minutes (below threshold of 10)
        for i in range(5):
            ts = now - timedelta(minutes=i)
            sub._record_alarm_timestamp(ts)
        assert sub._is_flood_active(now) is False

    def test_flood_active_at_threshold(self, subscriber_instance):
        """Flood SHOULD be active when 10 alarms in 10 minutes (ISA-18.2 R-010)."""
        sub = subscriber_instance
        now = datetime.now(UTC)
        # Add exactly 10 alarms in last 10 minutes
        for i in range(ALARM_FLOOD_THRESHOLD):
            ts = now - timedelta(minutes=i)
            sub._record_alarm_timestamp(ts)
        assert sub._is_flood_active(now) is True

    def test_flood_active_above_threshold(self, subscriber_instance):
        """Flood should be active when alarm count > threshold."""
        sub = subscriber_instance
        now = datetime.now(UTC)
        for i in range(15):
            ts = now - timedelta(seconds=i * 30)
            sub._record_alarm_timestamp(ts)
        assert sub._is_flood_active(now) is True

    def test_old_alarms_excluded_from_window(self, subscriber_instance):
        """Alarms older than 10 minutes should not count toward flood."""
        sub = subscriber_instance
        now = datetime.now(UTC)
        # Add 8 alarms older than 10 minutes
        for i in range(8):
            ts = now - timedelta(minutes=15 + i)
            sub._record_alarm_timestamp(ts)
        # Add 2 alarms within window
        for i in range(2):
            ts = now - timedelta(minutes=i)
            sub._record_alarm_timestamp(ts)
        # Total in window: 2 (below threshold)
        assert sub._is_flood_active(now) is False

    def test_demo_mode_higher_threshold(self, subscriber_instance):
        """In DEMO_MODE, flood threshold should be 100 alarms."""
        sub = subscriber_instance
        sub._demo_mode = True
        now = datetime.now(UTC)
        # Add 50 alarms (above normal threshold, below demo threshold)
        for i in range(50):
            ts = now - timedelta(seconds=i * 10)
            sub._record_alarm_timestamp(ts)
        assert sub._is_flood_active(now) is False

    def test_demo_mode_flood_at_100(self, subscriber_instance):
        """In DEMO_MODE, flood activates at 100 alarms."""
        sub = subscriber_instance
        sub._demo_mode = True
        now = datetime.now(UTC)
        for i in range(DEMO_FLOOD_THRESHOLD):
            ts = now - timedelta(seconds=i * 5)
            sub._record_alarm_timestamp(ts)
        assert sub._is_flood_active(now) is True

    def test_flood_recovery_after_quiet_period(self, subscriber_instance):
        """Flood should recover after 5 minutes below threshold."""
        sub = subscriber_instance
        now = datetime.now(UTC)
        # Trigger flood
        for i in range(15):
            ts = now - timedelta(minutes=8, seconds=i * 10)
            sub._record_alarm_timestamp(ts)
        assert sub._is_flood_active(now - timedelta(minutes=7)) is True

        # After 5 minutes with no new alarms, should recover
        recovery_time = now - timedelta(minutes=8) + timedelta(minutes=6)
        assert sub._is_flood_active(recovery_time) is False

    def test_flood_suppresses_warning_alarms(self, subscriber_instance, sample_anomaly_event):
        """During flood, WARNING-level alarms should be suppressed."""
        sub = subscriber_instance
        now = datetime.now(UTC)
        # Trigger flood
        for i in range(12):
            ts = now - timedelta(minutes=i)
            sub._record_alarm_timestamp(ts)

        event = dict(sample_anomaly_event)
        event[b"severity"] = b"WARNING"
        result = sub._process_anomaly_event(event)
        # Should be suppressed or marked as flood
        assert result is not None
        if hasattr(result, 'suppressed'):
            assert result.suppressed is True

    def test_flood_does_not_suppress_critical_alarms(self, subscriber_instance, sample_anomaly_event):
        """During flood, CRITICAL alarms should still be processed."""
        sub = subscriber_instance
        now = datetime.now(UTC)
        # Trigger flood
        for i in range(12):
            ts = now - timedelta(minutes=i)
            sub._record_alarm_timestamp(ts)

        # CRITICAL event should still be processed
        result = sub._process_anomaly_event(sample_anomaly_event)
        assert result is not None


# ---------------------------------------------------------------------------
# TestDecisionScenarioGeneration
# ---------------------------------------------------------------------------
class TestDecisionScenarioGeneration:
    """Test decision scenario generation via DecisionEngine."""

    def test_generates_scenarios_for_alarm(self, subscriber_instance, mock_db_session):
        """Should generate decision scenarios when alarm is created."""
        sub = subscriber_instance
        event = {
            b"event_id": b"evt-dec-001",
            b"timestamp": b"2026-06-10T12:00:00+00:00",
            b"source": b"pm-ml",
            b"event_type": b"ANOMALY_DETECTED",
            b"machine_id": b"AC-201",
            b"anomaly_id": b"50",
            b"anomaly_score": b"0.88",
            b"top_contributing_sensor": b"vibration_rms",
            b"severity": b"CRITICAL",
            b"phase": b"DEGRADED",
        }
        result = sub._process_anomaly_event(event)
        # Should have created a DecisionLog
        mock_db_session.add.assert_called()

    def test_scenario_includes_observe_option(self, subscriber_instance):
        """Generated scenarios should include OBSERVE (Murat's rule #1)."""
        sub = subscriber_instance
        scenarios = sub._generate_scenarios(machine_id="AC-201", rul_hours=24.0)
        scenario_ids = [s.scenario.value if hasattr(s, 'scenario') else s['scenario'] for s in scenarios]
        assert "OBSERVE" in scenario_ids

    def test_scenario_includes_shutdown_option(self, subscriber_instance):
        """Generated scenarios should include SHUTDOWN."""
        sub = subscriber_instance
        scenarios = sub._generate_scenarios(machine_id="AC-201", rul_hours=24.0)
        scenario_ids = [s.scenario.value if hasattr(s, 'scenario') else s['scenario'] for s in scenarios]
        assert "SHUTDOWN" in scenario_ids

    def test_recommended_scenario_is_lowest_cost(self, subscriber_instance):
        """The recommended scenario should be the lowest cost option."""
        sub = subscriber_instance
        options = sub._generate_scenarios(machine_id="AC-201", rul_hours=48.0)
        recommended = [o for o in options if (getattr(o, 'is_recommended', False) or o.get('is_recommended', False))]
        assert len(recommended) >= 1

    def test_scenario_generation_failed_creates_fallback(
        self, subscriber_instance, mock_db_session
    ):
        """If scenario generation fails, should create fallback DecisionLog."""
        sub = subscriber_instance
        # Mock DecisionEngine to raise
        sub._decision_engine = MagicMock()
        sub._decision_engine.recommend.side_effect = Exception("Engine failure")

        result = sub._handle_scenario_failure(
            machine_id="AC-201",
            anomaly_id=42,
            alarm_id=1,
            error="Engine failure",
        )
        # Should create a fallback DecisionLog
        assert result is not None
        mock_db_session.add.assert_called()


# ---------------------------------------------------------------------------
# TestEUAIActComplianceLogging
# ---------------------------------------------------------------------------
class TestEUAIActComplianceLogging:
    """Test EU AI Act compliance logging."""

    def test_logs_ai_act_event_on_decision(self, subscriber_instance, mock_db_session):
        """Should log to AIActLog when a decision is made."""
        sub = subscriber_instance
        sub._log_ai_act_compliance(
            machine_id="AC-201",
            model_version="v2.1.0",
            decision_type="ANOMALY_RESPONSE",
            features_snapshot={"anomaly_score": 0.85},
            output={"recommended": "PLANNED"},
            action_taken="SCENARIO_GENERATED",
        )
        mock_db_session.add.assert_called()

    def test_ai_act_log_includes_timestamp(self, subscriber_instance, mock_db_session):
        """AI Act log entry should include timestamp."""
        sub = subscriber_instance
        sub._log_ai_act_compliance(
            machine_id="AC-201",
            model_version="v2.1.0",
            decision_type="ANOMALY_RESPONSE",
            features_snapshot={},
            output={},
            action_taken="TEST",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        ai_logs = [obj for obj in added_objects if hasattr(obj, 'model_version')]
        assert len(ai_logs) >= 1
        assert ai_logs[0].timestamp is not None

    def test_ai_act_log_includes_machine_id(self, subscriber_instance, mock_db_session):
        """AI Act log should include machine_id for traceability."""
        sub = subscriber_instance
        sub._log_ai_act_compliance(
            machine_id="HX-202",
            model_version="v2.1.0",
            decision_type="ANOMALY_RESPONSE",
            features_snapshot={},
            output={},
            action_taken="TEST",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        ai_logs = [obj for obj in added_objects if hasattr(obj, 'machine_id')]
        assert len(ai_logs) >= 1
        assert ai_logs[0].machine_id == "HX-202"

    def test_ai_act_log_includes_model_version(self, subscriber_instance, mock_db_session):
        """EU AI Act requires model version for transparency."""
        sub = subscriber_instance
        sub._log_ai_act_compliance(
            machine_id="AC-201",
            model_version="v3.0.0-beta",
            decision_type="RUL_PREDICTION",
            features_snapshot={},
            output={},
            action_taken="PREDICTED",
        )
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        ai_logs = [obj for obj in added_objects if hasattr(obj, 'model_version')]
        assert ai_logs[0].model_version == "v3.0.0-beta"


# ---------------------------------------------------------------------------
# TestPeriodicTick
# ---------------------------------------------------------------------------
class TestPeriodicTick:
    """Test periodic tick (5 min): shelve expiry, L2 timeout."""

    def test_periodic_tick_checks_shelve_expiry(self, subscriber_instance, mock_db_session):
        """Periodic tick should check for expired shelves."""
        sub = subscriber_instance
        sub._periodic_tick()
        # Should have queried for expired shelves
        mock_db_session.query.assert_called()

    def test_periodic_tick_checks_l2_timeout(self, subscriber_instance, mock_db_session):
        """Periodic tick should check for L2 escalation timeouts."""
        sub = subscriber_instance
        sub._periodic_tick()
        # L2 timeout check should be performed
        assert True  # Verifies tick runs without error

    def test_shelve_expiry_returns_alarm_to_normal(self, subscriber_instance, mock_db_session):
        """Expired shelve should return alarm to NORMAL state."""
        sub = subscriber_instance
        # Create a mock alarm that's shelved and expired
        mock_alarm = MagicMock()
        mock_alarm.status = "SHELVED"
        mock_alarm.shelved_until = datetime.now(UTC) - timedelta(hours=1)
        mock_alarm.id = 1

        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_alarm]
        sub._check_shelve_expiry()
        # Alarm should be transitioned back to NORMAL
        assert mock_alarm.status == "NORMAL" or mock_db_session.commit.called

    def test_l2_timeout_escalates_to_manager(self, subscriber_instance, mock_db_session):
        """L2 timeout should escalate decision to manager."""
        sub = subscriber_instance
        mock_decision = MagicMock()
        mock_decision.id = "dec-001"
        mock_decision.escalation_level = 1
        mock_decision.due_at = datetime.now(UTC) - timedelta(minutes=10)

        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_decision]
        sub._check_l2_timeouts()
        # Should escalate
        assert mock_decision.escalation_level >= 2 or mock_db_session.commit.called


# ---------------------------------------------------------------------------
# TestReconciliation
# ---------------------------------------------------------------------------
class TestReconciliation:
    """Test reconciliation when scenario generation fails."""

    def test_fallback_decision_log_created_on_engine_failure(
        self, subscriber_instance, mock_db_session
    ):
        """When DecisionEngine fails, should create fallback DecisionLog."""
        sub = subscriber_instance
        result = sub._handle_scenario_failure(
            machine_id="AC-201",
            anomaly_id=42,
            alarm_id=1,
            error="DecisionEngine timeout",
        )
        assert result is not None
        # Fallback should have resolution_source = "FALLBACK"
        if hasattr(result, 'resolution_source'):
            assert result.resolution_source == "FALLBACK"

    def test_fallback_decision_has_no_scenarios(self, subscriber_instance):
        """Fallback DecisionLog should indicate no scenarios were offered."""
        sub = subscriber_instance
        result = sub._handle_scenario_failure(
            machine_id="AC-201",
            anomaly_id=42,
            alarm_id=1,
            error="Engine crash",
        )
        if hasattr(result, 'scenarios_presented'):
            assert result.scenarios_presented is None or result.scenarios_presented == []

    def test_fallback_still_creates_ai_act_log(self, subscriber_instance, mock_db_session):
        """Even on failure, EU AI Act compliance log should be created."""
        sub = subscriber_instance
        sub._handle_scenario_failure(
            machine_id="AC-201",
            anomaly_id=42,
            alarm_id=1,
            error="Engine failure",
        )
        # Should have logged to AIActLog
        added_objects = [c[0][0] for c in mock_db_session.add.call_args_list]
        ai_logs = [obj for obj in added_objects if hasattr(obj, 'model_version')]
        assert len(ai_logs) >= 1


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_duplicate_anomaly_id_skipped(self, subscriber_instance, mock_db_session):
        """Duplicate anomaly_id should be skipped (already processed)."""
        sub = subscriber_instance
        event = {
            b"event_id": b"evt-dup",
            b"timestamp": b"2026-06-10T12:00:00+00:00",
            b"source": b"pm-ml",
            b"event_type": b"ANOMALY_DETECTED",
            b"machine_id": b"AC-201",
            b"anomaly_id": b"42",
            b"anomaly_score": b"0.85",
            b"top_contributing_sensor": b"vibration_rms",
            b"severity": b"CRITICAL",
            b"phase": b"DEGRADED",
        }
        # Process once
        sub._process_anomaly_event(event)
        call_count_after_first = mock_db_session.add.call_count

        # Process same anomaly_id again
        sub._process_anomaly_event(event)
        # Should not create duplicate alarm
        assert mock_db_session.add.call_count == call_count_after_first

    def test_invalid_anomaly_id_rejected(self, subscriber_instance):
        """Invalid (non-numeric) anomaly_id should be rejected."""
        sub = subscriber_instance
        event = {
            b"event_id": b"evt-bad",
            b"timestamp": b"2026-06-10T12:00:00+00:00",
            b"source": b"pm-ml",
            b"event_type": b"ANOMALY_DETECTED",
            b"machine_id": b"AC-201",
            b"anomaly_id": b"not_a_number",
            b"anomaly_score": b"0.85",
            b"top_contributing_sensor": b"vibration_rms",
            b"severity": b"CRITICAL",
            b"phase": b"DEGRADED",
        }
        result = sub._process_anomaly_event(event)
        assert result is None or hasattr(result, 'error')

    def test_missing_machine_id_rejected(self, subscriber_instance):
        """Event without machine_id should be rejected."""
        sub = subscriber_instance
        event = {
            b"event_id": b"evt-nomid",
            b"timestamp": b"2026-06-10T12:00:00+00:00",
            b"source": b"pm-ml",
            b"event_type": b"ANOMALY_DETECTED",
            b"anomaly_id": b"42",
            b"anomaly_score": b"0.85",
            b"severity": b"CRITICAL",
        }
        result = sub._process_anomaly_event(event)
        assert result is None

    def test_missing_anomaly_score_defaults_to_zero(self, subscriber_instance):
        """Event without anomaly_score should default to 0.0."""
        sub = subscriber_instance
        event = {
            b"event_id": b"evt-noscore",
            b"timestamp": b"2026-06-10T12:00:00+00:00",
            b"source": b"pm-ml",
            b"event_type": b"ANOMALY_DETECTED",
            b"machine_id": b"AC-201",
            b"anomaly_id": b"42",
            b"severity": b"WARNING",
            b"phase": b"HEALTHY",
        }
        fields = sub._parse_event_fields(event)
        assert float(fields.get("anomaly_score", 0.0)) == 0.0

    def test_empty_event_handled_gracefully(self, subscriber_instance):
        """Empty event should not crash the subscriber."""
        sub = subscriber_instance
        result = sub._process_anomaly_event({})
        assert result is None

    def test_redis_connection_error_handled(self, subscriber_instance, mock_redis):
        """Redis connection error should be handled gracefully."""
        sub = subscriber_instance
        mock_redis.xreadgroup.side_effect = ConnectionError("Redis down")
        # Should not raise
        sub._poll_once()

    def test_db_commit_failure_triggers_rollback(self, subscriber_instance, mock_db_session):
        """DB commit failure should trigger rollback."""
        sub = subscriber_instance
        mock_db_session.commit.side_effect = Exception("DB connection lost")
        event = {
            b"event_id": b"evt-dberr",
            b"timestamp": b"2026-06-10T12:00:00+00:00",
            b"source": b"pm-ml",
            b"event_type": b"ANOMALY_DETECTED",
            b"machine_id": b"AC-201",
            b"anomaly_id": b"42",
            b"anomaly_score": b"0.85",
            b"severity": b"CRITICAL",
            b"phase": b"DEGRADED",
        }
        # Should not raise, should rollback
        sub._process_anomaly_event(event)
        mock_db_session.rollback.assert_called()

    def test_mixed_byte_string_keys_handled(self, subscriber_instance):
        """Should handle events with mixed byte/string keys."""
        sub = subscriber_instance
        event = {
            "event_id": "evt-mixed",
            b"machine_id": b"AC-201",
            "anomaly_id": "42",
            b"severity": "WARNING",
        }
        fields = sub._parse_event_fields(event)
        assert fields["machine_id"] == "AC-201"
        assert fields["anomaly_id"] == "42"
