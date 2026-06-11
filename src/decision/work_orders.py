"""
Work Order Manager (Phase 2C - Decision Engine Layer).

Manages work orders generated from decision options.

Rules:
  - OBSERVATION skip rule: No work order created for OBSERVATION type
  - create_from_option: Creates work orders for non-OBSERVATION types
  - Priority mapping: severity -> priority (HIGH->P1, MEDIUM->P2, LOW->P3)
  - Assigned_to mapping: type -> team
  - WorkOrder dataclass
  - DB operations (SQLite)
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.decision.recommendation_engine import WorkOrderType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class WorkOrderStatus(str, Enum):
    """Work order status."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class Priority(str, Enum):
    """Work order priority."""
    P1 = "P1"  # Critical
    P2 = "P2"  # High
    P3 = "P3"  # Normal


class Severity(str, Enum):
    """Severity levels (mirrors recommendation_engine)."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBSERVATION_SKIP_RULE: bool = True

PRIORITY_MAP: dict[Severity, Priority] = {
    Severity.HIGH: Priority.P1,
    Severity.MEDIUM: Priority.P2,
    Severity.LOW: Priority.P3,
}

TEAM_MAP: dict[WorkOrderType, str] = {
    WorkOrderType.TECHNICAL_DISPATCH: "maintenance",
    WorkOrderType.SLOWDOWN_ORDER: "operations",
    WorkOrderType.STOP_PREP_ORDER: "maintenance",
    WorkOrderType.CONTROLLED_STOP: "operations",
    WorkOrderType.OBSERVATION: "none",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class WorkOrder:
    """A work order generated from a decision option."""
    id: str
    work_order_type: WorkOrderType
    machine_id: str
    priority: Priority
    created_at: datetime
    status: WorkOrderStatus = WorkOrderStatus.OPEN
    assigned_to: str | None = None
    description: str = ""
    updated_at: datetime | None = None
    closed_at: datetime | None = None
    diagnosis_id: str | None = None
    option_id: str | None = None
    severity: Severity | None = None
    estimated_cost: float = 0.0
    metadata_json: str | None = None


# ---------------------------------------------------------------------------
# Work Order Manager
# ---------------------------------------------------------------------------
class WorkOrderManager:
    """
    Manages work orders from decision options.

    OBSERVATION type is skipped (no work order created).
    """

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create work_orders table if not exists."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS work_orders (
                id TEXT PRIMARY KEY,
                work_order_type TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                assigned_to TEXT,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                closed_at TEXT,
                diagnosis_id TEXT,
                option_id TEXT,
                severity TEXT,
                estimated_cost REAL,
                metadata_json TEXT
            )
        """)
        self.db.commit()

    def create_from_option(self, option: dict) -> WorkOrder | None:
        """
        Create a work order from a decision option.

        OBSERVATION type is skipped (returns None).

        Args:
            option: Dict with keys:
                - work_order_type: WorkOrderType
                - machine_id: str
                - severity: Severity
                - expected_cost: float
                - description: str
                - diagnosis_id: str
                - option_id: str

        Returns:
            WorkOrder or None (for OBSERVATION)
        """
        wo_type = option.get("work_order_type")

        # OBSERVATION skip rule
        if OBSERVATION_SKIP_RULE and wo_type == WorkOrderType.OBSERVATION:
            return None

        # Check for duplicate option_id
        option_id = option.get("option_id")
        if option_id:
            existing = self._get_by_option_id(option_id)
            if existing:
                return existing

        # Generate work order
        wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
        severity = option.get("severity", Severity.MEDIUM)
        priority = PRIORITY_MAP.get(severity, Priority.P3)
        assigned_to = TEAM_MAP.get(wo_type, "unassigned")
        now = datetime.utcnow()

        wo = WorkOrder(
            id=wo_id,
            work_order_type=wo_type,
            machine_id=option.get("machine_id", "UNKNOWN"),
            priority=priority,
            status=WorkOrderStatus.OPEN,
            assigned_to=assigned_to,
            description=option.get("description", ""),
            created_at=now,
            diagnosis_id=option.get("diagnosis_id"),
            option_id=option_id,
            severity=severity,
            estimated_cost=option.get("expected_cost", 0.0),
        )

        # Persist to DB
        self._insert(wo)

        return wo

    def create_batch(self, options: list[dict]) -> list[WorkOrder]:
        """Create work orders from multiple options, skipping OBSERVATION."""
        results = []
        for option in options:
            wo = self.create_from_option(option)
            if wo is not None:
                results.append(wo)
        return results

    def get_by_id(self, wo_id: str) -> WorkOrder | None:
        """Retrieve a work order by ID."""
        cursor = self.db.execute(
            "SELECT * FROM work_orders WHERE id = ?", (wo_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_work_order(row)

    def list_by_status(self, status: WorkOrderStatus) -> list[WorkOrder]:
        """List all work orders with the given status."""
        cursor = self.db.execute(
            "SELECT * FROM work_orders WHERE status = ?", (status.value,)
        )
        return [self._row_to_work_order(row) for row in cursor.fetchall()]

    def list_by_machine(self, machine_id: str) -> list[WorkOrder]:
        """List all work orders for a specific machine."""
        cursor = self.db.execute(
            "SELECT * FROM work_orders WHERE machine_id = ?", (machine_id,)
        )
        return [self._row_to_work_order(row) for row in cursor.fetchall()]

    def close(self, wo_id: str, closed_by: str = "") -> WorkOrder | None:
        """Close a work order."""
        now = datetime.utcnow()
        self.db.execute(
            "UPDATE work_orders SET status = ?, closed_at = ?, updated_at = ? WHERE id = ?",
            (WorkOrderStatus.CLOSED.value, now.isoformat(), now.isoformat(), wo_id)
        )
        self.db.commit()
        return self.get_by_id(wo_id)

    def update_status(self, wo_id: str, new_status: WorkOrderStatus) -> WorkOrder | None:
        """Update work order status."""
        now = datetime.utcnow()
        self.db.execute(
            "UPDATE work_orders SET status = ?, updated_at = ? WHERE id = ?",
            (new_status.value, now.isoformat(), wo_id)
        )
        self.db.commit()
        return self.get_by_id(wo_id)

    def _get_by_option_id(self, option_id: str) -> WorkOrder | None:
        """Check if a work order already exists for this option_id."""
        cursor = self.db.execute(
            "SELECT * FROM work_orders WHERE option_id = ?", (option_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_work_order(row)

    def _insert(self, wo: WorkOrder) -> None:
        """Insert a work order into the database."""
        self.db.execute(
            """INSERT INTO work_orders
            (id, work_order_type, machine_id, priority, status, assigned_to,
             description, created_at, diagnosis_id, option_id, severity,
             estimated_cost, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                wo.id,
                wo.work_order_type.value,
                wo.machine_id,
                wo.priority.value,
                wo.status.value,
                wo.assigned_to,
                wo.description,
                wo.created_at.isoformat(),
                wo.diagnosis_id,
                wo.option_id,
                wo.severity.value if wo.severity else None,
                wo.estimated_cost,
                wo.metadata_json,
            )
        )
        self.db.commit()

    def _row_to_work_order(self, row) -> WorkOrder:
        """Convert a database row to a WorkOrder object."""
        # row is a tuple from sqlite3
        columns = [
            "id", "work_order_type", "machine_id", "priority", "status",
            "assigned_to", "description", "created_at", "updated_at",
            "closed_at", "diagnosis_id", "option_id", "severity",
            "estimated_cost", "metadata_json"
        ]
        data = dict(zip(columns, row, strict=False))

        return WorkOrder(
            id=data["id"],
            work_order_type=WorkOrderType(data["work_order_type"]),
            machine_id=data["machine_id"],
            priority=Priority(data["priority"]),
            status=WorkOrderStatus(data["status"]),
            assigned_to=data.get("assigned_to"),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at") else None
            ),
            closed_at=(
                datetime.fromisoformat(data["closed_at"])
                if data.get("closed_at") else None
            ),
            diagnosis_id=data.get("diagnosis_id"),
            option_id=data.get("option_id"),
            severity=Severity(data["severity"]) if data.get("severity") else None,
            estimated_cost=data.get("estimated_cost", 0.0),
            metadata_json=data.get("metadata_json"),
        )
