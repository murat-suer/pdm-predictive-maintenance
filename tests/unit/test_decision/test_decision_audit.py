"""
Unit tests for src.decision.decision_audit (Phase 2C - Decision Engine Layer)

Tests DecisionAudit: Immutable audit trail (EU AI Act compliance)
  - AuditEvent dataclass
  - write_event: Immutable record (cannot be modified after write)
  - query_by_machine: Machine-based query
  - query_by_alarm: Alarm-based query
  - query_recent: Last N events
  - query_overrides: Operator override detection
  - JSON serialization (ml_inputs, diagnosis, options_offered)

NOTE: These tests will FAIL until decision_audit.py is implemented.
"""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.decision_audit import (
    EU_AI_ACT_COMPLIANT,
    IMMUTABILITY_ENFORCED,
    MAX_QUERY_LIMIT,
    AuditEvent,
    AuditEventType,
    AuditStore,
    DecisionAudit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def db_connection():
    """In-memory SQLite connection for audit store."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            alarm_id TEXT,
            timestamp TEXT NOT NULL,
            ml_inputs_json TEXT,
            diagnosis_json TEXT,
            options_offered_json TEXT,
            selected_option_id TEXT,
            operator_override BOOLEAN DEFAULT 0,
            override_reason TEXT,
            operator_id TEXT,
            cascade_cost REAL,
            survival_probability REAL,
            metadata_json TEXT
        )
    """)
    yield conn
    conn.close()


@pytest.fixture
def audit(db_connection):
    """DecisionAudit with in-memory SQLite store."""
    return DecisionAudit(store=AuditStore(db=db_connection))


@pytest.fixture
def sample_event():
    """Sample audit event."""
    return AuditEvent(
        id="audit-001",
        event_type=AuditEventType.DECISION_MADE,
        machine_id="AC-001",
        alarm_id="alarm-123",
        timestamp=datetime.utcnow(),
        ml_inputs={"vibration_rms": 4.5, "temperature": 72.0, "pressure": 3.8},
        diagnosis={"type": "PROCESS_ANOMALY", "mode": "bearing_outer_race", "rpn": 180},
        options_offered=[
            {"type": "OBSERVATION", "cost": 0.0, "recommended": True},
            {"type": "SLOWDOWN_ORDER", "cost": 2500.0, "recommended": False},
            {"type": "STOP_PREP_ORDER", "cost": 8000.0, "recommended": False},
        ],
        selected_option_id="opt-001",
        operator_override=False,
        operator_id=None,
        cascade_cost=3500.0,
        survival_probability=0.85,
    )


@pytest.fixture
def override_event():
    """Audit event with operator override."""
    return AuditEvent(
        id="audit-002",
        event_type=AuditEventType.DECISION_MADE,
        machine_id="AC-001",
        alarm_id="alarm-456",
        timestamp=datetime.utcnow(),
        ml_inputs={"vibration_rms": 6.0, "temperature": 80.0},
        diagnosis={"type": "PROCESS_ANOMALY", "mode": "bearing_inner_race", "rpn": 250},
        options_offered=[
            {"type": "STOP_PREP_ORDER", "cost": 8000.0, "recommended": True},
        ],
        selected_option_id="opt-override",
        operator_override=True,
        override_reason="Operator chose to continue production for 2 more hours",
        operator_id="operator_murat",
        cascade_cost=0.0,
        survival_probability=0.35,
    )


# ---------------------------------------------------------------------------
# TestAuditEventDataclass
# ---------------------------------------------------------------------------
class TestAuditEventDataclass:
    """AuditEvent dataclass validation."""

    def test_audit_event_has_required_fields(self, sample_event):
        """AuditEvent must have all required fields."""
        assert sample_event.id is not None
        assert sample_event.event_type is not None
        assert sample_event.machine_id is not None
        assert sample_event.timestamp is not None

    def test_audit_event_type_enum(self):
        """AuditEventType must have expected values."""
        assert AuditEventType.DECISION_MADE is not None
        assert AuditEventType.OPTION_SELECTED is not None
        assert AuditEventType.OVERRIDE is not None

    def test_audit_event_has_ml_inputs(self, sample_event):
        """AuditEvent should store ML inputs."""
        assert sample_event.ml_inputs is not None
        assert "vibration_rms" in sample_event.ml_inputs

    def test_audit_event_has_diagnosis(self, sample_event):
        """AuditEvent should store diagnosis."""
        assert sample_event.diagnosis is not None
        assert sample_event.diagnosis["type"] == "PROCESS_ANOMALY"

    def test_audit_event_has_options_offered(self, sample_event):
        """AuditEvent should store all offered options."""
        assert sample_event.options_offered is not None
        assert len(sample_event.options_offered) == 3

    def test_audit_event_timestamp_is_datetime(self, sample_event):
        """Timestamp should be a datetime object."""
        assert isinstance(sample_event.timestamp, datetime)


# ---------------------------------------------------------------------------
# TestWriteEvent - Immutability
# ---------------------------------------------------------------------------
class TestWriteEventImmutability:
    """write_event: Immutable record (EU AI Act compliance)."""

    def test_immutability_enforced(self):
        """IMMUTABILITY_ENFORCED must be True."""
        assert IMMUTABILITY_ENFORCED is True

    def test_eu_ai_act_compliant(self):
        """EU_AI_ACT_COMPLIANT must be True."""
        assert EU_AI_ACT_COMPLIANT is True

    def test_write_event_stores_record(self, audit, sample_event):
        """write_event should persist the event."""
        audit.write_event(sample_event)
        events = audit.query_by_machine("AC-001")
        assert len(events) == 1
        assert events[0].id == sample_event.id

    def test_written_event_cannot_be_modified(self, audit, sample_event):
        """After write, event fields cannot be changed."""
        audit.write_event(sample_event)
        # Attempt to modify should raise or be silently ignored
        with pytest.raises((AttributeError, RuntimeError, TypeError)):
            retrieved = audit.query_by_machine("AC-001")[0]
            retrieved.machine_id = "MODIFIED"

    def test_write_event_idempotent(self, audit, sample_event):
        """Writing the same event twice should not create duplicates."""
        audit.write_event(sample_event)
        audit.write_event(sample_event)  # same ID
        events = audit.query_by_machine("AC-001")
        assert len(events) == 1

    def test_write_event_preserves_all_fields(self, audit, sample_event):
        """All fields should be preserved after write."""
        audit.write_event(sample_event)
        events = audit.query_by_machine("AC-001")
        retrieved = events[0]
        assert retrieved.machine_id == sample_event.machine_id
        assert retrieved.event_type == sample_event.event_type
        assert retrieved.survival_probability == sample_event.survival_probability


# ---------------------------------------------------------------------------
# TestQueryByMachine
# ---------------------------------------------------------------------------
class TestQueryByMachine:
    """query_by_machine: Machine-based query."""

    def test_query_by_machine_returns_events(self, audit, sample_event):
        """Should return events for the specified machine."""
        audit.write_event(sample_event)
        events = audit.query_by_machine("AC-001")
        assert len(events) >= 1

    def test_query_by_machine_empty_for_unknown(self, audit, sample_event):
        """Should return empty list for unknown machine."""
        audit.write_event(sample_event)
        events = audit.query_by_machine("UNKNOWN-MACHINE")
        assert len(events) == 0

    def test_query_by_machine_filters_correctly(self, audit):
        """Should only return events for the queried machine."""
        event_ac = AuditEvent(
            id="audit-ac",
            event_type=AuditEventType.DECISION_MADE,
            machine_id="AC-001",
            timestamp=datetime.utcnow(),
            ml_inputs={},
            diagnosis={},
            options_offered=[],
        )
        event_hx = AuditEvent(
            id="audit-hx",
            event_type=AuditEventType.DECISION_MADE,
            machine_id="HX-001",
            timestamp=datetime.utcnow(),
            ml_inputs={},
            diagnosis={},
            options_offered=[],
        )
        audit.write_event(event_ac)
        audit.write_event(event_hx)

        ac_events = audit.query_by_machine("AC-001")
        assert all(e.machine_id == "AC-001" for e in ac_events)

        hx_events = audit.query_by_machine("HX-001")
        assert all(e.machine_id == "HX-001" for e in hx_events)

    def test_query_by_machine_returns_multiple(self, audit):
        """Should return all events for a machine."""
        for i in range(5):
            event = AuditEvent(
                id=f"audit-multi-{i}",
                event_type=AuditEventType.DECISION_MADE,
                machine_id="AC-001",
                timestamp=datetime.utcnow(),
                ml_inputs={},
                diagnosis={},
                options_offered=[],
            )
            audit.write_event(event)
        events = audit.query_by_machine("AC-001")
        assert len(events) == 5


# ---------------------------------------------------------------------------
# TestQueryByAlarm
# ---------------------------------------------------------------------------
class TestQueryByAlarm:
    """query_by_alarm: Alarm-based query."""

    def test_query_by_alarm_returns_events(self, audit, sample_event):
        """Should return events linked to the specified alarm."""
        audit.write_event(sample_event)
        events = audit.query_by_alarm("alarm-123")
        assert len(events) >= 1

    def test_query_by_alarm_empty_for_unknown(self, audit, sample_event):
        """Should return empty for unknown alarm."""
        audit.write_event(sample_event)
        events = audit.query_by_alarm("alarm-UNKNOWN")
        assert len(events) == 0

    def test_query_by_alarm_filters_correctly(self, audit):
        """Should only return events for the queried alarm."""
        event1 = AuditEvent(
            id="audit-a1",
            event_type=AuditEventType.DECISION_MADE,
            machine_id="AC-001",
            alarm_id="alarm-AAA",
            timestamp=datetime.utcnow(),
            ml_inputs={},
            diagnosis={},
            options_offered=[],
        )
        event2 = AuditEvent(
            id="audit-a2",
            event_type=AuditEventType.DECISION_MADE,
            machine_id="AC-001",
            alarm_id="alarm-BBB",
            timestamp=datetime.utcnow(),
            ml_inputs={},
            diagnosis={},
            options_offered=[],
        )
        audit.write_event(event1)
        audit.write_event(event2)

        aaa_events = audit.query_by_alarm("alarm-AAA")
        assert all(e.alarm_id == "alarm-AAA" for e in aaa_events)


# ---------------------------------------------------------------------------
# TestQueryRecent
# ---------------------------------------------------------------------------
class TestQueryRecent:
    """query_recent: Last N events."""

    def test_query_recent_returns_n_events(self, audit):
        """Should return the last N events."""
        for i in range(10):
            event = AuditEvent(
                id=f"audit-recent-{i}",
                event_type=AuditEventType.DECISION_MADE,
                machine_id="AC-001",
                timestamp=datetime.utcnow() + timedelta(minutes=i),
                ml_inputs={},
                diagnosis={},
                options_offered=[],
            )
            audit.write_event(event)

        recent = audit.query_recent(limit=5)
        assert len(recent) == 5

    def test_query_recent_ordered_by_timestamp(self, audit):
        """Recent events should be ordered by timestamp (newest first)."""
        for i in range(5):
            event = AuditEvent(
                id=f"audit-order-{i}",
                event_type=AuditEventType.DECISION_MADE,
                machine_id="AC-001",
                timestamp=datetime.utcnow() + timedelta(minutes=i),
                ml_inputs={},
                diagnosis={},
                options_offered=[],
            )
            audit.write_event(event)

        recent = audit.query_recent(limit=3)
        timestamps = [e.timestamp for e in recent]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_recent_default_limit(self, audit):
        """Default limit should be reasonable (e.g., 50 or 100)."""
        recent = audit.query_recent()
        # Should not exceed MAX_QUERY_LIMIT
        assert len(recent) <= MAX_QUERY_LIMIT

    def test_query_recent_respects_max_limit(self, audit):
        """Cannot request more than MAX_QUERY_LIMIT."""
        with pytest.raises((ValueError, TypeError)):
            audit.query_recent(limit=MAX_QUERY_LIMIT + 1000)


# ---------------------------------------------------------------------------
# TestQueryOverrides
# ---------------------------------------------------------------------------
class TestQueryOverrides:
    """query_overrides: Operator override detection."""

    def test_query_overrides_returns_override_events(self, audit, override_event):
        """Should return events where operator overrode the recommendation."""
        audit.write_event(override_event)
        overrides = audit.query_overrides()
        assert len(overrides) >= 1
        assert all(e.operator_override is True for e in overrides)

    def test_query_overrides_excludes_normal_events(self, audit, sample_event, override_event):
        """Should not include non-override events."""
        audit.write_event(sample_event)
        audit.write_event(override_event)
        overrides = audit.query_overrides()
        # Only override_event should appear
        assert len(overrides) == 1
        assert overrides[0].operator_override is True

    def test_override_event_has_reason(self, audit, override_event):
        """Override events should have a reason."""
        audit.write_event(override_event)
        overrides = audit.query_overrides()
        assert overrides[0].override_reason is not None
        assert len(overrides[0].override_reason) > 0

    def test_override_event_has_operator_id(self, audit, override_event):
        """Override events should record who overrode."""
        audit.write_event(override_event)
        overrides = audit.query_overrides()
        assert overrides[0].operator_id is not None
        assert overrides[0].operator_id == "operator_murat"

    def test_no_overrides_returns_empty(self, audit, sample_event):
        """No overrides → empty list."""
        audit.write_event(sample_event)
        overrides = audit.query_overrides()
        assert len(overrides) == 0


# ---------------------------------------------------------------------------
# TestJSONSerialization
# ---------------------------------------------------------------------------
class TestJSONSerialization:
    """JSON serialization for ml_inputs, diagnosis, options_offered."""

    def test_ml_inputs_serializable(self, sample_event):
        """ml_inputs should be JSON-serializable."""
        json_str = json.dumps(sample_event.ml_inputs)
        parsed = json.loads(json_str)
        assert parsed == sample_event.ml_inputs

    def test_diagnosis_serializable(self, sample_event):
        """diagnosis should be JSON-serializable."""
        json_str = json.dumps(sample_event.diagnosis)
        parsed = json.loads(json_str)
        assert parsed == sample_event.diagnosis

    def test_options_offered_serializable(self, sample_event):
        """options_offered should be JSON-serializable."""
        json_str = json.dumps(sample_event.options_offered)
        parsed = json.loads(json_str)
        assert parsed == sample_event.options_offered
        assert len(parsed) == 3

    def test_roundtrip_preserves_data(self, audit, sample_event):
        """Write → Read → JSON should preserve all data."""
        audit.write_event(sample_event)
        events = audit.query_by_machine("AC-001")
        retrieved = events[0]

        # Serialize and deserialize
        ml_json = json.dumps(retrieved.ml_inputs)
        diag_json = json.dumps(retrieved.diagnosis)
        opts_json = json.dumps(retrieved.options_offered)

        assert json.loads(ml_json) == sample_event.ml_inputs
        assert json.loads(diag_json) == sample_event.diagnosis
        assert json.loads(opts_json) == sample_event.options_offered

    def test_complex_ml_inputs_serializable(self):
        """Complex nested ML inputs should serialize correctly."""
        complex_inputs = {
            "vibration_rms": {"value": 4.5, "trend": "rising", "baseline": 2.3},
            "temperature": {"value": 72.0, "trend": "stable", "baseline": 65.0},
            "harmonics": [0.8, 1.2, 0.3, 0.1],
            "envelope_spectrum": {"bpfo": 0.9, "bpfi": 0.2, "bsf": 0.1},
        }
        json_str = json.dumps(complex_inputs)
        parsed = json.loads(json_str)
        assert parsed["vibration_rms"]["value"] == 4.5
        assert len(parsed["harmonics"]) == 4


# ---------------------------------------------------------------------------
# TestAuditStore
# ---------------------------------------------------------------------------
class TestAuditStore:
    """AuditStore: Low-level storage operations."""

    def test_store_creates_table(self, db_connection):
        """AuditStore should create the audit_events table."""
        store = AuditStore(db=db_connection)
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'"
        )
        assert cursor.fetchone() is not None

    def test_store_insert_and_retrieve(self, db_connection, sample_event):
        """Store should insert and retrieve events."""
        store = AuditStore(db=db_connection)
        store.insert(sample_event)
        events = store.query(machine_id="AC-001")
        assert len(events) == 1

    def test_store_count(self, db_connection):
        """Store should report correct event count."""
        store = AuditStore(db=db_connection)
        for i in range(3):
            event = AuditEvent(
                id=f"audit-count-{i}",
                event_type=AuditEventType.DECISION_MADE,
                machine_id="AC-001",
                timestamp=datetime.utcnow(),
                ml_inputs={},
                diagnosis={},
                options_offered=[],
            )
            store.insert(event)
        assert store.count() == 3


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestAuditEdgeCases:
    """Edge cases for the audit system."""

    def test_event_with_no_alarm_id(self, audit):
        """Events without alarm_id should still be stored."""
        event = AuditEvent(
            id="audit-no-alarm",
            event_type=AuditEventType.DECISION_MADE,
            machine_id="AC-001",
            alarm_id=None,
            timestamp=datetime.utcnow(),
            ml_inputs={},
            diagnosis={},
            options_offered=[],
        )
        audit.write_event(event)
        events = audit.query_by_machine("AC-001")
        assert len(events) == 1

    def test_event_with_empty_options(self, audit):
        """Events with empty options_offered should be stored."""
        event = AuditEvent(
            id="audit-empty-opts",
            event_type=AuditEventType.DECISION_MADE,
            machine_id="AC-001",
            timestamp=datetime.utcnow(),
            ml_inputs={},
            diagnosis={},
            options_offered=[],
        )
        audit.write_event(event)
        events = audit.query_by_machine("AC-001")
        assert len(events[0].options_offered) == 0

    def test_event_with_null_survival_probability(self, audit):
        """Events with None survival_probability should be handled."""
        event = AuditEvent(
            id="audit-null-surv",
            event_type=AuditEventType.DECISION_MADE,
            machine_id="AC-001",
            timestamp=datetime.utcnow(),
            ml_inputs={},
            diagnosis={},
            options_offered=[],
            survival_probability=None,
        )
        audit.write_event(event)
        events = audit.query_by_machine("AC-001")
        assert events[0].survival_probability is None

    def test_concurrent_writes(self, audit):
        """Multiple rapid writes should not lose data."""
        events_written = []
        for i in range(20):
            event = AuditEvent(
                id=f"audit-concurrent-{i}",
                event_type=AuditEventType.DECISION_MADE,
                machine_id="AC-001",
                timestamp=datetime.utcnow(),
                ml_inputs={},
                diagnosis={},
                options_offered=[],
            )
            audit.write_event(event)
            events_written.append(event.id)

        all_events = audit.query_by_machine("AC-001")
        assert len(all_events) == 20
