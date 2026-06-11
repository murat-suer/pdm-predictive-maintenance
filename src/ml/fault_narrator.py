"""Compose evidence-driven diagnoses from confirmed simulated faults.

Phase 3A adds the "honest UNKNOWN" path: when the fault classifier's
confidence is below `DIAGNOSIS_UNKNOWN_CONFIDENCE`, the narrator
returns a diagnosis with `fault_type=UNKNOWN` and a single
`DISPATCH_TECHNICIAN` recommended action. This is a deliberate design
choice: the system says "I don't know" rather than fabricating a
fault from low-confidence ML evidence. The decision engine reads the
diagnosis and surfaces a `DISPATCH_TECHNICIAN` scenario with the
appropriate cost and required role.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DIAGNOSIS_UNKNOWN_CONFIDENCE = 0.50
DIAGNOSIS_UNKNOWN_FAULT_TYPE = "UNKNOWN"
DIAGNOSIS_DISPATCH_ACTION = "DISPATCH_TECHNICIAN"


@dataclass(frozen=True)
class Evidence:
    sensor: str
    current: float
    unit: str
    baseline_mean: float
    baseline_sigma: float
    threshold_used: float
    deviation_pct: float
    shap_contribution: float
    narrative: str


@dataclass(frozen=True)
class RuledOut:
    fault_type: str
    reason: str


@dataclass(frozen=True)
class RecommendedAction:
    action: str
    expected_outcome: str | None = None
    part_no: str | None = None
    lead_time_days: int | None = None
    window: str | None = None
    estimated_downtime_min: int | None = None
    cost_eur: float | None = None


@dataclass(frozen=True)
class FaultDiagnosis:
    machine_id: str
    fault_type: str
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    ruled_out: list[RuledOut] = field(default_factory=list)
    probable_cause: dict[str, Any] = field(default_factory=dict)
    rul: dict[str, Any] = field(default_factory=dict)
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    estimated_failure_cost_eur_per_hour: float = 0.0

    @property
    def is_unknown(self) -> bool:
        """True when the narrator is honestly saying "I don't know"."""
        return self.fault_type == DIAGNOSIS_UNKNOWN_FAULT_TYPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compose_diagnosis(
    machine_id: str,
    confirmed_fault: dict[str, Any],
    payload: dict[str, Any],
    baselines: dict[str, tuple[float, float]],
    shap_values: dict[str, float],
    rul_estimate: dict[str, Any] | None,
    upstream_state: str | None,
    cost_per_hour: float,
    unknown_confidence_threshold: float = DIAGNOSIS_UNKNOWN_CONFIDENCE,
) -> FaultDiagnosis:
    """Compose a diagnostic explanation from ML and physics evidence.

    If `confirmed_fault["confidence"]` is below
    `unknown_confidence_threshold` (default 0.50), the diagnosis is
    flagged UNKNOWN with a single `DISPATCH_TECHNICIAN` recommended
    action. The system prefers an honest "send a human" to a fabricated
    classification that would mislead the operator.
    """
    confidence = float(confirmed_fault.get("confidence", 0.0))
    if confidence < unknown_confidence_threshold:
        return _compose_unknown_diagnosis(
            machine_id=machine_id,
            confidence=confidence,
            cost_per_hour=cost_per_hour,
            rul_estimate=rul_estimate,
        )

    fault_type = confirmed_fault["fault_type"]
    return FaultDiagnosis(
        machine_id=machine_id,
        fault_type=fault_type,
        confidence=confidence,
        evidence=_evidence(payload, baselines, shap_values),
        ruled_out=_ruled_out(fault_type, payload, upstream_state),
        probable_cause=_probable_cause(fault_type, payload),
        rul=dict(rul_estimate or {}),
        recommended_actions=_recommended_actions(fault_type, rul_estimate or {}),
        estimated_failure_cost_eur_per_hour=cost_per_hour,
    )


def _compose_unknown_diagnosis(
    machine_id: str,
    confidence: float,
    cost_per_hour: float,
    rul_estimate: dict[str, Any] | None,
) -> FaultDiagnosis:
    """
    Build a diagnosis that honestly says "UNKNOWN — send a technician".

    The narrative is an i18n key so the dashboard can localise it
    (Turkish / English / German). The `recommended_actions` list has a
    single `DISPATCH_TECHNICIAN` action so the decision engine can pick
    it up verbatim when constructing scenarios.
    """
    rul = dict(rul_estimate or {})
    return FaultDiagnosis(
        machine_id=machine_id,
        fault_type=DIAGNOSIS_UNKNOWN_FAULT_TYPE,
        confidence=round(confidence, 4),
        evidence=[],
        ruled_out=[],
        probable_cause={
            "key": "probable_cause_unknown",
            "params": {"confidence": f"{confidence:.0%}"},
        },
        rul=rul,
        recommended_actions=[
            RecommendedAction(
                action=DIAGNOSIS_DISPATCH_ACTION,
                expected_outcome="Confirm the underlying fault via in-person inspection",
                window="next_handover",
                estimated_downtime_min=60,
                cost_eur=180.0,
            )
        ],
        estimated_failure_cost_eur_per_hour=cost_per_hour,
    )


def _evidence(
    payload: dict[str, Any],
    baselines: dict[str, tuple[float, float]],
    shap_values: dict[str, float],
) -> list[Evidence]:
    rows = []
    for sensor, (mean, sigma) in baselines.items():
        value = _as_float(payload.get(sensor))
        if value is None or sigma <= 0:
            continue
        z_score = (value - mean) / sigma
        if abs(z_score) < 3.0:
            continue
        shap = float(shap_values.get(f"{sensor}_value", shap_values.get(sensor, 0.0)) or 0.0)
        rows.append(
            Evidence(
                sensor=sensor,
                current=value,
                unit=_unit_for(sensor),
                baseline_mean=mean,
                baseline_sigma=sigma,
                threshold_used=mean + (3 * sigma if z_score >= 0 else -3 * sigma),
                deviation_pct=100.0 * (value - mean) / max(abs(mean), 1e-6),
                shap_contribution=shap,
                narrative=(
                    f"{sensor} {value:.2f} {_unit_for(sensor)} "
                    f"vs baseline {mean:.2f}+/-{sigma:.2f} ({z_score:+.2f} sigma)"
                ),
            )
        )
    rows.sort(key=lambda row: (abs(row.shap_contribution), abs(row.deviation_pct)), reverse=True)
    return rows[:5]


def _ruled_out(fault_type: str, payload: dict[str, Any], upstream_state: str | None) -> list[RuledOut]:
    rows = []
    oil_pressure = _as_float(payload.get("oil_pressure"))
    if fault_type == "BEARING_FAULT" and oil_pressure is not None and 3.5 <= oil_pressure <= 5.0:
        rows.append(RuledOut("OIL_STARVATION", f"oil_pressure {oil_pressure:.2f} bar remains nominal"))
    flow_rate = _as_float(payload.get("flow_rate"))
    if fault_type == "FOULING" and flow_rate is not None and flow_rate > 7.0:
        rows.append(RuledOut("TOTAL_FLOW_LOSS", f"flow_rate {flow_rate:.2f} m3/h still shows throughput"))
    if upstream_state == "NORMAL":
        rows.append(RuledOut("UPSTREAM_CASCADE", "upstream machine state is NORMAL"))
    return rows


def _probable_cause(fault_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return an i18n key + params dict. The dashboard resolves via _ti18n()."""
    if fault_type == "BEARING_FAULT":
        return {"key": "probable_cause_bearing", "params": {"value": f"{_metric(payload, 'vibration_kurtosis'):.2f}"}}
    if fault_type == "OIL_DEGRADATION":
        return {"key": "probable_cause_oil", "params": {"value": f"{_metric(payload, 'oil_pressure'):.2f}"}}
    if fault_type == "FOULING":
        return {"key": "probable_cause_fouling", "params": {"value": f"{_metric(payload, 'fouling_indicator'):.4f}"}}
    if fault_type == "BELT_SLIP":
        return {"key": "probable_cause_belt", "params": {"value": f"{_metric(payload, 'belt_slip'):.1%}"}}
    return {"key": "probable_cause_unclassified"}


def _recommended_actions(fault_type: str, rul: dict[str, Any]) -> list[RecommendedAction]:
    low_rul = rul.get("rul_low_ci") or rul.get("p10_hours") or rul.get("rul_hours") or 24
    window = f"next_{max(float(low_rul), 24.0):.0f}h"
    actions = {
        "BEARING_FAULT": [
            RecommendedAction("REDUCE_LOAD_20PCT", expected_outcome="Slow vibration and thermal drift"),
            RecommendedAction("PLAN_BEARING_INSPECTION", window=window, estimated_downtime_min=90, cost_eur=450),
        ],
        "OIL_DEGRADATION": [
            RecommendedAction("TAKE_OIL_SAMPLE", expected_outcome="Confirm leak or lubricant degradation"),
            RecommendedAction("SCHEDULE_OIL_SERVICE", window=window, estimated_downtime_min=45, cost_eur=180),
        ],
        "FOULING": [
            RecommendedAction("CHECK_PRESSURE_DROP_AND_FLOW", expected_outcome="Confirm deposit growth"),
            RecommendedAction("PLAN_CIP_CYCLE", window=window, estimated_downtime_min=120, cost_eur=220),
        ],
        "BELT_SLIP": [
            RecommendedAction("CHECK_BELT_TENSION", expected_outcome="Restore traction if tension drifted"),
            RecommendedAction("INSPECT_PULLEY_AND_BELT", window=window, estimated_downtime_min=30, cost_eur=85),
        ],
    }
    return actions.get(fault_type, [RecommendedAction("MONITOR_AND_TRIAGE", window=window)])


def _unit_for(sensor: str) -> str:
    return {
        "vibration_rms": "mm/s",
        "bearing_temp": "C",
        "oil_pressure": "bar",
        "pressure_drop": "bar",
        "flow_rate": "m3/h",
        "belt_tension": "kN",
        "speed_rpm": "RPM",
        "drive_temp": "C",
    }.get(sensor, "")


def _metric(payload: dict[str, Any], key: str) -> float:
    return _as_float(payload.get(key)) or 0.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
