"""
Decision Audit (Phase 2C - Decision Engine Layer).

Immutable audit trail for EU AI Act compliance.

Features:
  - AuditEvent dataclass
  - write_event: Immutable record (cannot be modified after write)
  - query_by_machine: Machine-based query
  - query_by_alarm: Alarm-based query
  - query_recent: Last N events
  - query_overrides: Operator override detection
  - JSON serialization (ml_inputs, diagnosis, options_offered)
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMMUTABILITY_ENFORCED: bool = True
EU_AI_ACT_COMPLIANT: bool = True
MAX_QUERY_LIMIT: int = 1000


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AuditEventType(str, Enum):
    """Types of audit events."""
    DECISION_MADE = "DECISION_MADE"
    OPTION_SELECTED = "OPTION_SELECTED"
    OVERRIDE = "OVERRIDE"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AuditEvent:
    """
    An immutable audit event recording a decision.

    After write_event(), the returned object is frozen (immutable).
    """
    id: str
    event_type: AuditEventType
    machine_id: str
    timestamp: datetime
    ml_inputs: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    options_offered: list[dict[str, Any]] = field(default_factory=list)
    alarm_id: str | None = None
    selected_option_id: str | None = None
    operator_override: bool = False
    override_reason: str | None = None
    operator_id: str | None = None
    cascade_cost: float | None = None
    survival_probability: float | None = None
    metadata_json: str | None = None


def _freeze_event(event: AuditEvent) -> AuditEvent:
    """
    Return a frozen (immutable) copy of the event.
    Attempting to set attributes on the frozen copy will raise AttributeError.
    """
    # Create a new instance with same data
    frozen = AuditEvent(
        id=event.id,
        event_type=event.event_type,
        machine_id=event.machine_id,
        timestamp=event.timestamp,
        ml_inputs=dict(event.ml_inputs) if event.ml_inputs else {},
        diagnosis=dict(event.diagnosis) if event.diagnosis else {},
        options_offered=list(event.options_offered) if event.options_offered else [],
        alarm_id=event.alarm_id,
        selected_option_id=event.selected_option_id,
        operator_override=event.operator_override,
        override_reason=event.override_reason,
        operator_id=event.operator_id,
        cascade_cost=event.cascade_cost,
        survival_probability=event.survival_probability,
        metadata_json=event.metadata_json,
    )
    # Make immutable by overriding __setattr__
    frozen.__class__ = _FrozenAuditEvent
    return frozen


class _FrozenAuditEvent(AuditEvent):
    """Immutable version of AuditEvent - raises on attribute assignment."""

    def __setattr__(self, name, value):
        raise AttributeError(
            f"AuditEvent is immutable after write. Cannot set '{name}'."
        )

    def __delattr__(self, name):
        raise AttributeError(
            f"AuditEvent is immutable after write. Cannot delete '{name}'."
        )


# ---------------------------------------------------------------------------
# Audit Store (low-level storage)
# ---------------------------------------------------------------------------
class AuditStore:
    """Low-level SQLite storage for audit events."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create audit_events table if not exists."""
        self.db.execute("""
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
        self.db.commit()

    def insert(self, event: AuditEvent) -> None:
        """Insert an audit event (idempotent - ignores duplicates)."""
        try:
            self.db.execute(
                """INSERT OR IGNORE INTO audit_events
                (id, event_type, machine_id, alarm_id, timestamp,
                 ml_inputs_json, diagnosis_json, options_offered_json,
                 selected_option_id, operator_override, override_reason,
                 operator_id, cascade_cost, survival_probability, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.id,
                    event.event_type.value,
                    event.machine_id,
                    event.alarm_id,
                    event.timestamp.isoformat(),
                    json.dumps(event.ml_inputs) if event.ml_inputs else "{}",
                    json.dumps(event.diagnosis) if event.diagnosis else "{}",
                    json.dumps(event.options_offered) if event.options_offered else "[]",
                    event.selected_option_id,
                    1 if event.operator_override else 0,
                    event.override_reason,
                    event.operator_id,
                    event.cascade_cost,
                    event.survival_probability,
                    event.metadata_json,
                )
            )
            self.db.commit()
        except sqlite3.IntegrityError:
            # Duplicate ID - idempotent
            pass

    def query(
        self,
        machine_id: str | None = None,
        alarm_id: str | None = None,
        operator_override: bool | None = None,
        limit: int = 100,
        order_by: str = "timestamp DESC",
    ) -> list[AuditEvent]:
        """Query events with optional filters."""
        conditions = []
        params = []

        if machine_id is not None:
            conditions.append("machine_id = ?")
            params.append(machine_id)

        if alarm_id is not None:
            conditions.append("alarm_id = ?")
            params.append(alarm_id)

        if operator_override is not None:
            conditions.append("operator_override = ?")
            params.append(1 if operator_override else 0)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"SELECT * FROM audit_events {where_clause} ORDER BY {order_by} LIMIT ?"
        params.append(limit)

        cursor = self.db.execute(sql, params)
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Return total event count."""
        cursor = self.db.execute("SELECT COUNT(*) FROM audit_events")
        return cursor.fetchone()[0]

    def _row_to_event(self, row) -> AuditEvent:
        """Convert a database row to an AuditEvent."""
        columns = [
            "id", "event_type", "machine_id", "alarm_id", "timestamp",
            "ml_inputs_json", "diagnosis_json", "options_offered_json",
            "selected_option_id", "operator_override", "override_reason",
            "operator_id", "cascade_cost", "survival_probability", "metadata_json"
        ]
        data = dict(zip(columns, row, strict=False))

        return AuditEvent(
            id=data["id"],
            event_type=AuditEventType(data["event_type"]),
            machine_id=data["machine_id"],
            alarm_id=data.get("alarm_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ml_inputs=json.loads(data["ml_inputs_json"] or "{}"),
            diagnosis=json.loads(data["diagnosis_json"] or "{}"),
            options_offered=json.loads(data["options_offered_json"] or "[]"),
            selected_option_id=data.get("selected_option_id"),
            operator_override=bool(data.get("operator_override", 0)),
            override_reason=data.get("override_reason"),
            operator_id=data.get("operator_id"),
            cascade_cost=data.get("cascade_cost"),
            survival_probability=data.get("survival_probability"),
            metadata_json=data.get("metadata_json"),
        )


# ---------------------------------------------------------------------------
# Decision Audit (high-level API)
# ---------------------------------------------------------------------------
class DecisionAudit:
    """
    High-level audit API with immutability enforcement.

    EU AI Act compliant: all decision records are immutable.
    """

    def __init__(self, store: AuditStore):
        self.store = store

    def write_event(self, event: AuditEvent) -> None:
        """
        Write an audit event (immutable, idempotent).

        After writing, the event cannot be modified.
        Writing the same event ID twice is a no-op.
        """
        self.store.insert(event)

    def query_by_machine(self, machine_id: str) -> list[AuditEvent]:
        """Query all events for a specific machine. Returns frozen events."""
        events = self.store.query(machine_id=machine_id, limit=MAX_QUERY_LIMIT)
        if IMMUTABILITY_ENFORCED:
            return [_freeze_event(e) for e in events]
        return events

    def query_by_alarm(self, alarm_id: str) -> list[AuditEvent]:
        """Query all events linked to a specific alarm. Returns frozen events."""
        events = self.store.query(alarm_id=alarm_id, limit=MAX_QUERY_LIMIT)
        if IMMUTABILITY_ENFORCED:
            return [_freeze_event(e) for e in events]
        return events

    def query_recent(self, limit: int = 50) -> list[AuditEvent]:
        """
        Query the most recent N events (newest first).

        Raises ValueError if limit > MAX_QUERY_LIMIT.
        """
        if limit > MAX_QUERY_LIMIT:
            raise ValueError(
                f"Limit {limit} exceeds MAX_QUERY_LIMIT ({MAX_QUERY_LIMIT})"
            )
        events = self.store.query(limit=limit, order_by="timestamp DESC")
        if IMMUTABILITY_ENFORCED:
            return [_freeze_event(e) for e in events]
        return events

    def query_overrides(self) -> list[AuditEvent]:
        """Query all events where operator overrode the recommendation."""
        events = self.store.query(operator_override=True, limit=MAX_QUERY_LIMIT)
        if IMMUTABILITY_ENFORCED:
            return [_freeze_event(e) for e in events]
        return events
