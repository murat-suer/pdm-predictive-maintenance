
from src.ml.fault_narrator import (
    DIAGNOSIS_DISPATCH_ACTION,
    DIAGNOSIS_UNKNOWN_FAULT_TYPE,
    compose_diagnosis,
)


def _bearing_fault(confidence=0.85):
    return {"fault_type": "BEARING_FAULT", "confidence": confidence}


def _baselines():
    return {
        "vibration_rms": (2.0, 0.3),
        "bearing_temp": (55.0, 2.0),
        "oil_pressure": (4.2, 0.3),
    }


def _payload_bearing():
    return {
        "vibration_rms": 8.5,
        "bearing_temp": 72.0,
        "oil_pressure": 4.2,
        "vibration_kurtosis": 5.8,
        "flow_rate": 8.5,
    }


def _shap():
    return {"vibration_rms_value": 0.7, "bearing_temp_value": 0.3}


class TestUnknownPath:
    def test_low_confidence_returns_unknown(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(confidence=0.30),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        assert diag.is_unknown
        assert diag.fault_type == DIAGNOSIS_UNKNOWN_FAULT_TYPE
        assert diag.confidence == 0.30

    def test_unknown_has_dispatch_action(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(confidence=0.10),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        assert len(diag.recommended_actions) == 1
        assert diag.recommended_actions[0].action == DIAGNOSIS_DISPATCH_ACTION

    def test_threshold_boundary_at_0_50(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(confidence=0.49),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        assert diag.is_unknown

    def test_at_threshold_not_unknown(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(confidence=0.50),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        assert not diag.is_unknown
        assert diag.fault_type == "BEARING_FAULT"


class TestEvidenceOrdering:
    def test_evidence_sorted_by_shap_then_deviation(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        assert len(diag.evidence) >= 1
        shap_vals = [abs(e.shap_contribution) for e in diag.evidence]
        assert shap_vals == sorted(shap_vals, reverse=True)

    def test_max_five_evidence_items(self):
        big_baselines = {f"sensor_{i}": (10.0, 0.1) for i in range(10)}
        big_payload = {f"sensor_{i}": 100.0 for i in range(10)}
        big_shap = {f"sensor_{i}_value": 0.1 * i for i in range(10)}
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(),
            payload=big_payload,
            baselines=big_baselines,
            shap_values=big_shap,
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        assert len(diag.evidence) <= 5


class TestRuledOut:
    def test_bearing_rules_out_oil_starvation_when_pressure_nominal(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        ruled_types = [r.fault_type for r in diag.ruled_out]
        assert "OIL_STARVATION" in ruled_types

    def test_normal_upstream_rules_out_cascade(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state="NORMAL",
            cost_per_hour=500.0,
        )
        ruled_types = [r.fault_type for r in diag.ruled_out]
        assert "UPSTREAM_CASCADE" in ruled_types

    def test_fouling_rules_out_total_flow_loss_when_flow_ok(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault={"fault_type": "FOULING", "confidence": 0.80},
            payload={"fouling_indicator": 0.05, "flow_rate": 8.5},
            baselines={},
            shap_values={},
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        ruled_types = [r.fault_type for r in diag.ruled_out]
        assert "TOTAL_FLOW_LOSS" in ruled_types


class TestDiagnosisStructure:
    def test_to_dict_returns_dict(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate={"rul_hours": 120},
            upstream_state=None,
            cost_per_hour=500.0,
        )
        d = diag.to_dict()
        assert isinstance(d, dict)
        assert d["machine_id"] == "AC-201"
        assert d["fault_type"] == "BEARING_FAULT"

    def test_cost_per_hour_propagated(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=999.0,
        )
        assert diag.estimated_failure_cost_eur_per_hour == 999.0

    def test_bearing_has_recommended_actions(self):
        diag = compose_diagnosis(
            machine_id="AC-201",
            confirmed_fault=_bearing_fault(),
            payload=_payload_bearing(),
            baselines=_baselines(),
            shap_values=_shap(),
            rul_estimate=None,
            upstream_state=None,
            cost_per_hour=500.0,
        )
        assert len(diag.recommended_actions) >= 1
        action_types = [a.action for a in diag.recommended_actions]
        assert "REDUCE_LOAD_20PCT" in action_types
