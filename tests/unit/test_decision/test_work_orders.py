"""
Unit tests for src.decision.work_orders (Phase 2C - Decision Engine Layer)

Tests WorkOrderManager:
  - OBSERVATION skip rule: No work order created for OBSERVATION type
  - create_from_option: Creates work orders for non-OBSERVATION types
  - Priority mapping: severity → priority (HIGH→P1, MEDIUM→P2, LOW→P3)
  - Assigned_to mapping: type → team
  - WorkOrder dataclass
  - DB operations (SQLite in-memory)

NOTE: These tests will FAIL until work_orders.py is implemented.
"""

import sqlite3
from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.work_orders import (
    OBSERVATION_SKIP_RULE,
    PRIORITY_MAP,
    TEAM_MAP,
    Priority,
    Severity,
    WorkOrder,
    WorkOrderManager,
    WorkOrderStatus,
    WorkOrderType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def db_connection():
    """In-memory SQLite connection for testing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
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
    yield conn
    conn.close()


@pytest.fixture
def manager(db_connection):
    """WorkOrderManager with in-memory SQLite."""
    return WorkOrderManager(db=db_connection)


@pytest.fixture
def observation_option():
    """OBSERVATION option (should be skipped)."""
    return {
        "work_order_type": WorkOrderType.OBSERVATION,
        "machine_id": "AC-001",
        "severity": Severity.LOW,
        "expected_cost": 0.0,
        "description": "Continue monitoring for 30 minutes",
        "diagnosis_id": "diag-001",
        "option_id": "opt-001",
    }


@pytest.fixture
def technical_dispatch_option():
    """TECHNICAL_DISPATCH option."""
    return {
        "work_order_type": WorkOrderType.TECHNICAL_DISPATCH,
        "machine_id": "AC-001",
        "severity": Severity.MEDIUM,
        "expected_cost": 15000.0,
        "description": "Send technician for inspection",
        "diagnosis_id": "diag-002",
        "option_id": "opt-002",
    }


@pytest.fixture
def slowdown_option():
    """SLOWDOWN_ORDER option."""
    return {
        "work_order_type": WorkOrderType.SLOWDOWN_ORDER,
        "machine_id": "AC-001",
        "severity": Severity.MEDIUM,
        "expected_cost": 2500.0,
        "description": "Reduce load by 20%",
        "diagnosis_id": "diag-003",
        "option_id": "opt-003",
    }


@pytest.fixture
def stop_prep_option():
    """STOP_PREP_ORDER option."""
    return {
        "work_order_type": WorkOrderType.STOP_PREP_ORDER,
        "machine_id": "AC-001",
        "severity": Severity.HIGH,
        "expected_cost": 8000.0,
        "description": "Prepare for controlled stop within 4 hours",
        "diagnosis_id": "diag-004",
        "option_id": "opt-004",
    }


@pytest.fixture
def controlled_stop_option():
    """CONTROLLED_STOP option."""
    return {
        "work_order_type": WorkOrderType.CONTROLLED_STOP,
        "machine_id": "AC-001",
        "severity": Severity.HIGH,
        "expected_cost": 25000.0,
        "description": "Execute controlled stop immediately",
        "diagnosis_id": "diag-005",
        "option_id": "opt-005",
    }


# ---------------------------------------------------------------------------
# TestOBSERVATIONSkipRule - KRİTİK
# ---------------------------------------------------------------------------
class TestOBSERVATIONSkipRule:
    """
    KRİTİK: OBSERVATION type must NEVER generate a work order.
    Observation is free (0 TL) and requires no physical action.
    """

    def test_observation_skip_rule_constant(self):
        """OBSERVATION_SKIP_RULE must be True."""
        assert OBSERVATION_SKIP_RULE is True

    def test_observation_does_not_create_work_order(self, manager, observation_option):
        """create_from_option with OBSERVATION returns None / skips."""
        result = manager.create_from_option(observation_option)
        assert result is None

    def test_observation_not_in_database(self, manager, observation_option, db_connection):
        """OBSERVATION should not appear in the database."""
        manager.create_from_option(observation_option)
        cursor = db_connection.execute(
            "SELECT COUNT(*) FROM work_orders WHERE work_order_type = ?",
            (WorkOrderType.OBSERVATION.value,)
        )
        count = cursor.fetchone()[0]
        assert count == 0

    def test_observation_no_side_effects(self, manager, observation_option, db_connection):
        """OBSERVATION should not trigger any DB writes."""
        cursor_before = db_connection.execute("SELECT COUNT(*) FROM work_orders")
        count_before = cursor_before.fetchone()[0]

        manager.create_from_option(observation_option)

        cursor_after = db_connection.execute("SELECT COUNT(*) FROM work_orders")
        count_after = cursor_after.fetchone()[0]
        assert count_before == count_after

    def test_batch_create_skips_observations(self, manager):
        """Batch create should skip OBSERVATION options."""
        options = [
            {
                "work_order_type": WorkOrderType.OBSERVATION,
                "machine_id": "AC-001",
                "severity": Severity.LOW,
                "expected_cost": 0.0,
                "description": "Monitor",
                "diagnosis_id": "diag-001",
                "option_id": "opt-001",
            },
            {
                "work_order_type": WorkOrderType.SLOWDOWN_ORDER,
                "machine_id": "AC-001",
                "severity": Severity.MEDIUM,
                "expected_cost": 2500.0,
                "description": "Reduce load",
                "diagnosis_id": "diag-001",
                "option_id": "opt-002",
            },
        ]
        results = manager.create_batch(options)
        # Only 1 work order created (SLOWDOWN_ORDER), OBSERVATION skipped
        assert len(results) == 1
        assert results[0].work_order_type == WorkOrderType.SLOWDOWN_ORDER


# ---------------------------------------------------------------------------
# TestCreateFromOption
# ---------------------------------------------------------------------------
class TestCreateFromOption:
    """create_from_option: Creates work orders for non-OBSERVATION types."""

    def test_technical_dispatch_creates_work_order(self, manager, technical_dispatch_option):
        """TECHNICAL_DISPATCH should create a work order."""
        wo = manager.create_from_option(technical_dispatch_option)
        assert wo is not None
        assert wo.work_order_type == WorkOrderType.TECHNICAL_DISPATCH

    def test_slowdown_creates_work_order(self, manager, slowdown_option):
        """SLOWDOWN_ORDER should create a work order."""
        wo = manager.create_from_option(slowdown_option)
        assert wo is not None
        assert wo.work_order_type == WorkOrderType.SLOWDOWN_ORDER

    def test_stop_prep_creates_work_order(self, manager, stop_prep_option):
        """STOP_PREP_ORDER should create a work order."""
        wo = manager.create_from_option(stop_prep_option)
        assert wo is not None
        assert wo.work_order_type == WorkOrderType.STOP_PREP_ORDER

    def test_controlled_stop_creates_work_order(self, manager, controlled_stop_option):
        """CONTROLLED_STOP should create a work order."""
        wo = manager.create_from_option(controlled_stop_option)
        assert wo is not None
        assert wo.work_order_type == WorkOrderType.CONTROLLED_STOP

    def test_work_order_has_unique_id(self, manager, technical_dispatch_option, slowdown_option):
        """Each work order should have a unique ID."""
        wo1 = manager.create_from_option(technical_dispatch_option)
        wo2 = manager.create_from_option(slowdown_option)
        assert wo1.id != wo2.id

    def test_work_order_has_timestamp(self, manager, technical_dispatch_option):
        """Work order should have creation timestamp."""
        wo = manager.create_from_option(technical_dispatch_option)
        assert wo.created_at is not None
        assert isinstance(wo.created_at, datetime)

    def test_work_order_status_is_open(self, manager, technical_dispatch_option):
        """New work order should have OPEN status."""
        wo = manager.create_from_option(technical_dispatch_option)
        assert wo.status == WorkOrderStatus.OPEN

    def test_work_order_persisted_in_db(self, manager, technical_dispatch_option, db_connection):
        """Work order should be persisted in database."""
        wo = manager.create_from_option(technical_dispatch_option)
        cursor = db_connection.execute(
            "SELECT id FROM work_orders WHERE id = ?", (wo.id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == wo.id


# ---------------------------------------------------------------------------
# TestPriorityMapping
# ---------------------------------------------------------------------------
class TestPriorityMapping:
    """Priority mapping: severity → priority."""

    def test_high_severity_maps_to_p1(self):
        """HIGH severity → P1 (Critical)."""
        assert PRIORITY_MAP[Severity.HIGH] == Priority.P1

    def test_medium_severity_maps_to_p2(self):
        """MEDIUM severity → P2 (High)."""
        assert PRIORITY_MAP[Severity.MEDIUM] == Priority.P2

    def test_low_severity_maps_to_p3(self):
        """LOW severity → P3 (Normal)."""
        assert PRIORITY_MAP[Severity.LOW] == Priority.P3

    def test_work_order_gets_correct_priority_high(self, manager, stop_prep_option):
        """STOP_PREP_ORDER (HIGH severity) should get P1 priority."""
        wo = manager.create_from_option(stop_prep_option)
        assert wo.priority == Priority.P1

    def test_work_order_gets_correct_priority_medium(self, manager, slowdown_option):
        """SLOWDOWN_ORDER (MEDIUM severity) should get P2 priority."""
        wo = manager.create_from_option(slowdown_option)
        assert wo.priority == Priority.P2

    def test_work_order_gets_correct_priority_low(self, manager):
        """Low severity option should get P3 priority."""
        option = {
            "work_order_type": WorkOrderType.TECHNICAL_DISPATCH,
            "machine_id": "AC-001",
            "severity": Severity.LOW,
            "expected_cost": 5000.0,
            "description": "Inspect sensor",
            "diagnosis_id": "diag-006",
            "option_id": "opt-006",
        }
        wo = manager.create_from_option(option)
        assert wo.priority == Priority.P3


# ---------------------------------------------------------------------------
# TestAssignedToMapping
# ---------------------------------------------------------------------------
class TestAssignedToMapping:
    """Assigned_to mapping: WorkOrderType → team."""

    def test_technical_dispatch_assigned_to_maintenance(self, manager, technical_dispatch_option):
        """TECHNICAL_DISPATCH → maintenance team."""
        wo = manager.create_from_option(technical_dispatch_option)
        assert wo.assigned_to == TEAM_MAP[WorkOrderType.TECHNICAL_DISPATCH]

    def test_slowdown_assigned_to_operations(self, manager, slowdown_option):
        """SLOWDOWN_ORDER → operations team."""
        wo = manager.create_from_option(slowdown_option)
        assert wo.assigned_to == TEAM_MAP[WorkOrderType.SLOWDOWN_ORDER]

    def test_stop_prep_assigned_to_maintenance(self, manager, stop_prep_option):
        """STOP_PREP_ORDER → maintenance team."""
        wo = manager.create_from_option(stop_prep_option)
        assert wo.assigned_to == TEAM_MAP[WorkOrderType.STOP_PREP_ORDER]

    def test_controlled_stop_assigned_to_operations(self, manager, controlled_stop_option):
        """CONTROLLED_STOP → operations team."""
        wo = manager.create_from_option(controlled_stop_option)
        assert wo.assigned_to == TEAM_MAP[WorkOrderType.CONTROLLED_STOP]

    def test_all_types_have_team_mapping(self):
        """All non-OBSERVATION types must have a team mapping."""
        for wtype in WorkOrderType:
            if wtype != WorkOrderType.OBSERVATION:
                assert wtype in TEAM_MAP, f"{wtype} missing from TEAM_MAP"


# ---------------------------------------------------------------------------
# TestWorkOrderDataclass
# ---------------------------------------------------------------------------
class TestWorkOrderDataclass:
    """WorkOrder dataclass validation."""

    def test_work_order_fields(self):
        """WorkOrder should have all required fields."""
        wo = WorkOrder(
            id="WO-001",
            work_order_type=WorkOrderType.SLOWDOWN_ORDER,
            machine_id="AC-001",
            priority=Priority.P2,
            status=WorkOrderStatus.OPEN,
            assigned_to="maintenance",
            description="Reduce load by 20%",
            created_at=datetime.utcnow(),
        )
        assert wo.id == "WO-001"
        assert wo.machine_id == "AC-001"
        assert wo.priority == Priority.P2

    def test_work_order_default_status_is_open(self):
        """Default status should be OPEN."""
        wo = WorkOrder(
            id="WO-002",
            work_order_type=WorkOrderType.TECHNICAL_DISPATCH,
            machine_id="AC-001",
            priority=Priority.P1,
            created_at=datetime.utcnow(),
        )
        assert wo.status == WorkOrderStatus.OPEN

    def test_work_order_has_machine_id(self):
        """WorkOrder must reference a machine."""
        wo = WorkOrder(
            id="WO-003",
            work_order_type=WorkOrderType.STOP_PREP_ORDER,
            machine_id="HX-001",
            priority=Priority.P1,
            created_at=datetime.utcnow(),
        )
        assert wo.machine_id == "HX-001"


# ---------------------------------------------------------------------------
# TestDBOperations
# ---------------------------------------------------------------------------
class TestDBOperations:
    """Database operations: CRUD on work orders."""

    def test_get_work_order_by_id(self, manager, technical_dispatch_option, db_connection):
        """Should retrieve work order by ID."""
        wo = manager.create_from_option(technical_dispatch_option)
        retrieved = manager.get_by_id(wo.id)
        assert retrieved is not None
        assert retrieved.id == wo.id

    def test_get_nonexistent_returns_none(self, manager):
        """Getting non-existent ID returns None."""
        result = manager.get_by_id("NONEXISTENT-ID")
        assert result is None

    def test_list_open_work_orders(self, manager, technical_dispatch_option, slowdown_option):
        """Should list all OPEN work orders."""
        manager.create_from_option(technical_dispatch_option)
        manager.create_from_option(slowdown_option)
        open_orders = manager.list_by_status(WorkOrderStatus.OPEN)
        assert len(open_orders) == 2

    def test_list_by_machine(self, manager, technical_dispatch_option, db_connection):
        """Should list work orders for a specific machine."""
        manager.create_from_option(technical_dispatch_option)
        orders = manager.list_by_machine("AC-001")
        assert len(orders) >= 1
        assert all(o.machine_id == "AC-001" for o in orders)

    def test_close_work_order(self, manager, technical_dispatch_option):
        """Should be able to close a work order."""
        wo = manager.create_from_option(technical_dispatch_option)
        closed = manager.close(wo.id, closed_by="operator_1")
        assert closed.status == WorkOrderStatus.CLOSED
        assert closed.closed_at is not None

    def test_update_work_order_status(self, manager, technical_dispatch_option):
        """Should be able to update work order status."""
        wo = manager.create_from_option(technical_dispatch_option)
        updated = manager.update_status(wo.id, WorkOrderStatus.IN_PROGRESS)
        assert updated.status == WorkOrderStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# TestWorkOrderStatusEnum
# ---------------------------------------------------------------------------
class TestWorkOrderStatusEnum:
    """WorkOrderStatus enum validation."""

    def test_open_status_exists(self):
        """OPEN status must exist."""
        assert WorkOrderStatus.OPEN is not None

    def test_in_progress_status_exists(self):
        """IN_PROGRESS status must exist."""
        assert WorkOrderStatus.IN_PROGRESS is not None

    def test_closed_status_exists(self):
        """CLOSED status must exist."""
        assert WorkOrderStatus.CLOSED is not None

    def test_cancelled_status_exists(self):
        """CANCELLED status must exist."""
        assert WorkOrderStatus.CANCELLED is not None


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases for work order management."""

    def test_duplicate_option_id_rejected(self, manager, technical_dispatch_option):
        """Same option_id should not create duplicate work orders."""
        wo1 = manager.create_from_option(technical_dispatch_option)
        wo2 = manager.create_from_option(technical_dispatch_option)
        # Either rejected or same ID returned
        if wo2 is not None:
            assert wo1.id == wo2.id

    def test_empty_description_handled(self, manager):
        """Work order with empty description should still be created."""
        option = {
            "work_order_type": WorkOrderType.TECHNICAL_DISPATCH,
            "machine_id": "AC-001",
            "severity": Severity.MEDIUM,
            "expected_cost": 10000.0,
            "description": "",
            "diagnosis_id": "diag-007",
            "option_id": "opt-007",
        }
        wo = manager.create_from_option(option)
        assert wo is not None

    def test_zero_cost_non_observation(self, manager):
        """Non-OBSERVATION with 0 cost should still create work order."""
        option = {
            "work_order_type": WorkOrderType.SLOWDOWN_ORDER,
            "machine_id": "AC-001",
            "severity": Severity.LOW,
            "expected_cost": 0.0,
            "description": "Minimal slowdown",
            "diagnosis_id": "diag-008",
            "option_id": "opt-008",
        }
        wo = manager.create_from_option(option)
        assert wo is not None
        assert wo.work_order_type == WorkOrderType.SLOWDOWN_ORDER
