"""Shared Pydantic response models for the dashboard API."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MachineSummary(BaseModel):
    """Fleet-level summary of a single machine."""
    id: str
    name: str
    type: str
    line: str
    status: str  # normal | warning | critical | offline
    health_score: float | None = None
    rul_hours: float | None = None
    reliability: float | None = None
    classification: str | None = None
    top_alarm: str | None = None
    health_history: list[float] = []


class FleetSummary(BaseModel):
    """KPI counts across the fleet.

    Status vocabulary: normal | watch | action | critical | maintenance | offline.
    ``online``  = total − offline  (convenience total for connected machines).
    ``warning`` = watch + action   (backward-compatible aggregate for existing clients).
    """
    total: int
    online: int
    normal: int
    watch: int
    action: int
    warning: int  # = watch + action, kept for backward compatibility
    critical: int
    maintenance: int
    offline: int
    avg_reliability: float | None = None
    active_alarms: int


class HealthTrendPoint(BaseModel):
    """One aggregated fleet-health sample."""
    bucket: datetime
    avg_health_score: float


class SensorSnapshot(BaseModel):
    """Latest state and recent history of one sensor."""
    sensor_name: str
    unit: str | None = None
    value: float | None = None
    timestamp: datetime | None = None
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    nominal_mu: float | None = None
    nominal_sigma: float | None = None
    degradation_direction: int | None = None
    is_anomaly: bool = False
    history: list[float] = []


class MachineDetail(BaseModel):
    """Full detail payload for the machine page."""
    id: str
    name: str
    type: str
    line: str
    status: str
    standard: str | None = None
    failure_mode: str | None = None
    health_score: float | None = None
    rul_hours: float | None = None
    reliability: float | None = None
    availability: float | None = None
    condition: float | None = None
    classification: str | None = None
    confidence: float | None = None
    sensors: list[SensorSnapshot] = []
    active_faults: list[dict[str, Any]] = []


class AlarmItem(BaseModel):
    """Active alarm joined with its anomaly context."""
    id: int
    machine_id: str
    status: str
    level: int
    severity: str
    fault_type: str | None = None
    top_contributing_sensor: str | None = None
    anomaly_score: float | None = None
    created_at: datetime
    duration_minutes: int


class DecisionScenarioItem(BaseModel):
    """One scenario option presented to the operator."""
    scenario: str
    cost: float
    expected_cost: float = 0.0
    failure_probability: float = 0.0
    is_recommended: bool = False


class PendingDecision(BaseModel):
    """A decision awaiting operator action."""
    id: str
    machine_id: str
    alarm_id: int | None = None
    severity: str | None = None
    fault_type: str | None = None
    anomaly_score: float | None = None
    shap_values: dict[str, float] | None = None
    rul_hours: float | None = None
    ai_recommendation: str | None = None
    scenarios: list[DecisionScenarioItem] = []
    created_at: datetime
    due_at: datetime | None = None


class DecisionResolveRequest(BaseModel):
    """Operator action on a pending decision."""
    scenario_id: str
    operator_role: str = "OPERATOR"
    operator_id: str | None = None


class DecisionResolveResponse(BaseModel):
    """Result of resolving a decision."""
    id: str
    action: str
    chosen_scenario_id: str | None = None
    overridden: bool = False
    alarm_status: str | None = None
    work_order_id: str | None = None


class AuditEvent(BaseModel):
    """Unified audit-trail event (decisions + alarm transitions)."""
    id: str
    timestamp: datetime
    category: str  # decision | alarm | system
    severity: str  # info | warning | critical
    actor: str
    action: str
    target: str
    details: str | None = None


class AuditPage(BaseModel):
    """Paginated audit events."""
    events: list[AuditEvent]
    total: int


class WorkOrderItem(BaseModel):
    """Work order summary."""
    id: str
    work_order_number: str | None = None
    machine_id: str
    fault_type: str | None = None
    recommended_action: str | None = None
    priority: str
    status: str
    estimated_cost_eur: float | None = None
    created_at: datetime


class ShiftReportItem(BaseModel):
    """Shift report summary."""
    id: int
    shift_type: str
    shift_start: datetime
    shift_end: datetime
    generated_at: datetime
    report_data: dict[str, Any]


class TimelineEvent(BaseModel):
    """One decision event in the machine's post-repair history."""
    at: datetime
    recommendation: str | None = None
    tier: str  # normal | watch | action | critical
    outcome: str | None = None
    decided_by: str | None = None


class MachineTimeline(BaseModel):
    """Decision history since the machine's last real repair, newest first."""
    machine_id: str
    repaired_at: datetime | None = None
    count: int
    events: list[TimelineEvent]
