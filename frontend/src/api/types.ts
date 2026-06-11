// API response types mirroring src/api/schemas.py

export type MachineStatus = 'normal' | 'warning' | 'critical' | 'maintenance' | 'offline'

export interface MachineSummary {
  id: string
  name: string
  type: string
  line: string
  status: MachineStatus
  health_score: number | null
  rul_hours: number | null
  reliability: number | null
  classification: string | null
  top_alarm: string | null
  health_history: number[]
}

export interface FleetSummary {
  total: number
  normal: number
  warning: number
  critical: number
  maintenance: number
  offline: number
  avg_reliability: number | null
  active_alarms: number
}

export interface HealthTrendPoint {
  bucket: string
  avg_health_score: number
}

export interface SensorSnapshot {
  sensor_name: string
  unit: string | null
  value: number | null
  timestamp: string | null
  warning_threshold: number | null
  critical_threshold: number | null
  nominal_mu: number | null
  nominal_sigma: number | null
  degradation_direction: number | null
  is_anomaly: boolean
  history: number[]
}

export interface ActiveFault {
  fault_type: string | null
  confidence: number | null
  severity: string
  top_contributing_sensor: string | null
  anomaly_score: number | null
  detected_at: string
}

export interface MachineDetailData {
  id: string
  name: string
  type: string
  line: string
  status: MachineStatus
  standard: string | null
  failure_mode: string | null
  health_score: number | null
  rul_hours: number | null
  reliability: number | null
  availability: number | null
  condition: number | null
  classification: string | null
  confidence: number | null
  sensors: SensorSnapshot[]
  active_faults: ActiveFault[]
}

export interface AlarmItem {
  id: number
  machine_id: string
  status: string
  level: number
  severity: 'WARNING' | 'CRITICAL'
  fault_type: string | null
  top_contributing_sensor: string | null
  anomaly_score: number | null
  created_at: string
  duration_minutes: number
}

export interface DecisionScenario {
  scenario: string
  cost: number
  expected_cost?: number
  failure_probability?: number
  is_recommended: boolean
}

export interface PendingDecision {
  id: string
  machine_id: string
  alarm_id: number | null
  severity: string | null
  fault_type: string | null
  anomaly_score: number | null
  shap_values: Record<string, number> | null
  rul_hours: number | null
  ai_recommendation: string | null
  scenarios: DecisionScenario[]
  created_at: string
  due_at: string | null
}

export interface DecisionResolveResult {
  id: string
  action: string
  chosen_scenario_id: string | null
  overridden: boolean
  alarm_status: string | null
  work_order_id: string | null
}

export interface AuditEvent {
  id: string
  timestamp: string
  category: 'decision' | 'alarm' | 'system'
  severity: 'info' | 'warning' | 'critical'
  actor: string
  action: string
  target: string
  details: string | null
}

export interface AuditPage {
  events: AuditEvent[]
  total: number
}

export interface WorkOrderItem {
  id: string
  work_order_number: string | null
  machine_id: string
  fault_type: string | null
  recommended_action: string | null
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  status: string
  estimated_cost_eur: number | null
  created_at: string
}

export interface ShiftReportItem {
  id: number
  shift_type: string
  shift_start: string
  shift_end: string
  generated_at: string
  report_data: Record<string, unknown>
}

export interface LiveMachine {
  id: string
  status: MachineStatus
  top_alarm: string | null
  health_score: number | null
  rul_hours: number | null
  reliability: number | null
}

export interface LiveSnapshot {
  type: 'snapshot'
  machines: LiveMachine[]
  pending_decisions: number
}

export interface LiveAnomaly {
  type: 'anomaly'
  event: Record<string, string>
}

export type LiveMessage = LiveSnapshot | LiveAnomaly

export interface SavingsEvent {
  machine_id: string
  scenario: string | null
  performed_at: string
  decided_by: string | null
  actual_cost_eur: number
  avoided_cost_eur: number | null
  savings_eur: number | null
  downtime_minutes: number | null
}

export interface SavingsSummary {
  events: SavingsEvent[]
  total_actual_eur: number
  total_avoided_eur: number
  total_savings_eur: number
  maintenance_count: number
  window_hours: number
}

export interface WhatIfResult {
  machine_id: string
  rul_hours: number
  defer_hours: number
  act_now_cost_eur: number
  run_to_failure_cost_eur: number
  failure_probability: number
  deferred_risk_eur: number
  expected_deferred_cost_eur: number
  net_benefit_of_acting_now_eur: number
  breakeven_hours: number | null
}

export interface SensorSeriesPoint {
  timestamp: string
  value: number
  is_anomaly: boolean
}

export interface SensorSeries {
  machine_id: string
  minutes: number
  series: Record<string, SensorSeriesPoint[]>
}
