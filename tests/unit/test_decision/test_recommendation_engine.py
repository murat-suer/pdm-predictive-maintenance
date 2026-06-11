"""
Unit tests for src.decision.recommendation_engine (Phase 2C - Decision Engine Layer)

Tests RecommendationEngine: generates DecisionEnvelope with 1-3 options
based on diagnosis type and severity.

Rules:
  - UNKNOWN diagnosis → 1 option (TECHNICAL_DISPATCH)
  - SENSOR_ANOMALY → 2 options (sensor inspection + monitoring)
  - PROCESS_ANOMALY high severity (RPN > 200) → 1 option (STOP_PREP_ORDER)
  - PROCESS_ANOMALY medium severity (100 <= RPN <= 200) → 3 options
  - PROCESS_ANOMALY low severity (RPN < 100) → 2 options
  - Recommended marker = lowest E[cost]
  - Anti-pattern: 3+ same option cannot be recommended (CONTROLLED_STOP exempt)
  - WorkOrderType: OBSERVATION, TECHNICAL_DISPATCH, SLOWDOWN_ORDER,
                   STOP_PREP_ORDER, CONTROLLED_STOP

NOTE: These tests will FAIL until recommendation_engine.py is implemented.
"""

from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.recommendation_engine import (
    ANTI_PATTERN_MAX_REPEATS,
    CONTROLLED_STOP_EXEMPT,
    MAX_OPTIONS,
    MIN_OPTIONS,
    RPN_HIGH_THRESHOLD,
    RPN_LOW_THRESHOLD,
    DecisionOption,
    DiagnosisType,
    RecommendationEngine,
    Severity,
    WorkOrderType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def engine():
    """Fresh RecommendationEngine."""
    return RecommendationEngine()


@pytest.fixture
def unknown_diagnosis():
    """UNKNOWN diagnosis result."""
    return {
        "diagnosis_type": DiagnosisType.UNKNOWN,
        "machine_id": "AC-001",
        "confidence": 0.4,
        "rpn": 0,
        "severity": Severity.LOW,
    }


@pytest.fixture
def sensor_anomaly_diagnosis():
    """SENSOR_ANOMALY diagnosis result."""
    return {
        "diagnosis_type": DiagnosisType.SENSOR_ANOMALY,
        "machine_id": "AC-001",
        "confidence": 0.7,
        "sensor_name": "vibration_rms",
        "recurring_count": 5,
        "rpn": 80,
        "severity": Severity.LOW,
    }


@pytest.fixture
def process_anomaly_high():
    """PROCESS_ANOMALY with high severity (RPN > 200)."""
    return {
        "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
        "machine_id": "AC-001",
        "confidence": 0.9,
        "mode_id": "bearing_outer_race",
        "rpn": 250,
        "severity": Severity.HIGH,
    }


@pytest.fixture
def process_anomaly_medium():
    """PROCESS_ANOMALY with medium severity (100 <= RPN <= 200)."""
    return {
        "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
        "machine_id": "AC-001",
        "confidence": 0.75,
        "mode_id": "bearing_inner_race",
        "rpn": 150,
        "severity": Severity.MEDIUM,
    }


@pytest.fixture
def process_anomaly_low():
    """PROCESS_ANOMALY with low severity (RPN < 100)."""
    return {
        "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
        "machine_id": "AC-001",
        "confidence": 0.6,
        "mode_id": "misalignment",
        "rpn": 70,
        "severity": Severity.LOW,
    }


# ---------------------------------------------------------------------------
# TestDecisionEnvelopeStructure
# ---------------------------------------------------------------------------
class TestDecisionEnvelopeStructure:
    """DecisionEnvelope: 1-3 options with recommended marker."""

    def test_envelope_has_1_to_3_options(self, engine, process_anomaly_medium):
        """DecisionEnvelope must contain between 1 and 3 options."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        assert MIN_OPTIONS <= len(envelope.options) <= MAX_OPTIONS

    def test_envelope_has_exactly_one_recommended(self, engine, process_anomaly_medium):
        """Exactly one option must be marked as recommended."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        recommended = [o for o in envelope.options if o.is_recommended]
        assert len(recommended) == 1

    def test_envelope_recommended_is_lowest_cost(self, engine, process_anomaly_medium):
        """Recommended option must have the lowest expected cost."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        recommended = [o for o in envelope.options if o.is_recommended][0]
        min_cost = min(o.expected_cost for o in envelope.options)
        assert recommended.expected_cost == min_cost

    def test_envelope_has_timestamp(self, engine, process_anomaly_medium):
        """Envelope should have a creation timestamp."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        assert envelope.timestamp is not None
        assert isinstance(envelope.timestamp, datetime)

    def test_envelope_has_machine_id(self, engine, process_anomaly_medium):
        """Envelope should reference the machine."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        assert envelope.machine_id == "AC-001"

    def test_envelope_options_have_unique_types(self, engine, process_anomaly_medium):
        """Options should have different WorkOrderTypes (anti-pattern rule)."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        types = [o.work_order_type for o in envelope.options]
        # All types should be unique (unless CONTROLLED_STOP exempt)
        non_exempt = [t for t in types if t != WorkOrderType.CONTROLLED_STOP]
        assert len(non_exempt) == len(set(non_exempt))


# ---------------------------------------------------------------------------
# TestUNKNOWNDiagnosis
# ---------------------------------------------------------------------------
class TestUNKNOWNDiagnosis:
    """UNKNOWN diagnosis → 1 option (TECHNICAL_DISPATCH)."""

    def test_unknown_produces_one_option(self, engine, unknown_diagnosis):
        """UNKNOWN → exactly 1 option."""
        envelope = engine.generate(diagnosis=unknown_diagnosis)
        assert len(envelope.options) == 1

    def test_unknown_option_is_technical_dispatch(self, engine, unknown_diagnosis):
        """UNKNOWN → TECHNICAL_DISPATCH work order type."""
        envelope = engine.generate(diagnosis=unknown_diagnosis)
        assert envelope.options[0].work_order_type == WorkOrderType.TECHNICAL_DISPATCH

    def test_unknown_option_is_recommended(self, engine, unknown_diagnosis):
        """Single option should be auto-recommended."""
        envelope = engine.generate(diagnosis=unknown_diagnosis)
        assert envelope.options[0].is_recommended is True

    def test_unknown_has_positive_cost(self, engine, unknown_diagnosis):
        """TECHNICAL_DISPATCH should have positive cost."""
        envelope = engine.generate(diagnosis=unknown_diagnosis)
        assert envelope.options[0].expected_cost > 0.0


# ---------------------------------------------------------------------------
# TestSENSORANOMALYDiagnosis
# ---------------------------------------------------------------------------
class TestSENSORANOMALYDiagnosis:
    """SENSOR_ANOMALY → 2 options (sensor inspection + monitoring)."""

    def test_sensor_anomaly_produces_two_options(self, engine, sensor_anomaly_diagnosis):
        """SENSOR_ANOMALY → exactly 2 options."""
        envelope = engine.generate(diagnosis=sensor_anomaly_diagnosis)
        assert len(envelope.options) == 2

    def test_sensor_anomaly_includes_observation(self, engine, sensor_anomaly_diagnosis):
        """SENSOR_ANOMALY should include OBSERVATION option."""
        envelope = engine.generate(diagnosis=sensor_anomaly_diagnosis)
        types = [o.work_order_type for o in envelope.options]
        assert WorkOrderType.OBSERVATION in types

    def test_sensor_anomaly_includes_technical_dispatch(self, engine, sensor_anomaly_diagnosis):
        """SENSOR_ANOMALY should include TECHNICAL_DISPATCH (sensor inspection)."""
        envelope = engine.generate(diagnosis=sensor_anomaly_diagnosis)
        types = [o.work_order_type for o in envelope.options]
        assert WorkOrderType.TECHNICAL_DISPATCH in types

    def test_sensor_anomaly_recommended_is_lower_cost(self, engine, sensor_anomaly_diagnosis):
        """Recommended should be the lower-cost option."""
        envelope = engine.generate(diagnosis=sensor_anomaly_diagnosis)
        recommended = [o for o in envelope.options if o.is_recommended][0]
        min_cost = min(o.expected_cost for o in envelope.options)
        assert recommended.expected_cost == min_cost


# ---------------------------------------------------------------------------
# TestPROCESSANOMALYHighSeverity
# ---------------------------------------------------------------------------
class TestPROCESSANOMALYHighSeverity:
    """PROCESS_ANOMALY high severity (RPN > 200) → 1 option."""

    def test_high_severity_produces_one_option(self, engine, process_anomaly_high):
        """High severity (RPN > 200) → exactly 1 option."""
        assert process_anomaly_high["rpn"] > RPN_HIGH_THRESHOLD
        envelope = engine.generate(diagnosis=process_anomaly_high)
        assert len(envelope.options) == 1

    def test_high_severity_is_stop_prep_order(self, engine, process_anomaly_high):
        """High severity → STOP_PREP_ORDER (prepare for controlled stop)."""
        envelope = engine.generate(diagnosis=process_anomaly_high)
        assert envelope.options[0].work_order_type == WorkOrderType.STOP_PREP_ORDER

    def test_high_severity_has_highest_cost(self, engine, process_anomaly_high):
        """STOP_PREP_ORDER should have significant cost."""
        envelope = engine.generate(diagnosis=process_anomaly_high)
        assert envelope.options[0].expected_cost > 0.0


# ---------------------------------------------------------------------------
# TestPROCESSANOMALYMediumSeverity
# ---------------------------------------------------------------------------
class TestPROCESSANOMALYMediumSeverity:
    """PROCESS_ANOMALY medium severity (100 <= RPN <= 200) → 3 options."""

    def test_medium_severity_produces_three_options(self, engine, process_anomaly_medium):
        """Medium severity → exactly 3 options."""
        assert RPN_LOW_THRESHOLD <= process_anomaly_medium["rpn"] <= RPN_HIGH_THRESHOLD
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        assert len(envelope.options) == 3

    def test_medium_severity_includes_observation(self, engine, process_anomaly_medium):
        """Medium severity should include OBSERVATION option."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        types = [o.work_order_type for o in envelope.options]
        assert WorkOrderType.OBSERVATION in types

    def test_medium_severity_includes_slowdown(self, engine, process_anomaly_medium):
        """Medium severity should include SLOWDOWN_ORDER option."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        types = [o.work_order_type for o in envelope.options]
        assert WorkOrderType.SLOWDOWN_ORDER in types

    def test_medium_severity_includes_stop_prep(self, engine, process_anomaly_medium):
        """Medium severity should include STOP_PREP_ORDER option."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        types = [o.work_order_type for o in envelope.options]
        assert WorkOrderType.STOP_PREP_ORDER in types

    def test_medium_severity_recommended_is_observation_or_slowdown(self, engine, process_anomaly_medium):
        """Recommended should be OBSERVATION (0 TL) or SLOWDOWN (lowest cost)."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        recommended = [o for o in envelope.options if o.is_recommended][0]
        # OBSERVATION = 0 TL should be recommended
        assert recommended.work_order_type in [
            WorkOrderType.OBSERVATION,
            WorkOrderType.SLOWDOWN_ORDER,
        ]


# ---------------------------------------------------------------------------
# TestPROCESSANOMALYLowSeverity
# ---------------------------------------------------------------------------
class TestPROCESSANOMALYLowSeverity:
    """PROCESS_ANOMALY low severity (RPN < 100) → 2 options."""

    def test_low_severity_produces_two_options(self, engine, process_anomaly_low):
        """Low severity (RPN < 100) → exactly 2 options."""
        assert process_anomaly_low["rpn"] < RPN_LOW_THRESHOLD
        envelope = engine.generate(diagnosis=process_anomaly_low)
        assert len(envelope.options) == 2

    def test_low_severity_includes_observation(self, engine, process_anomaly_low):
        """Low severity should include OBSERVATION."""
        envelope = engine.generate(diagnosis=process_anomaly_low)
        types = [o.work_order_type for o in envelope.options]
        assert WorkOrderType.OBSERVATION in types

    def test_low_severity_observation_is_recommended(self, engine, process_anomaly_low):
        """For low severity, OBSERVATION (0 TL) should be recommended."""
        envelope = engine.generate(diagnosis=process_anomaly_low)
        recommended = [o for o in envelope.options if o.is_recommended][0]
        assert recommended.work_order_type == WorkOrderType.OBSERVATION
        assert recommended.expected_cost == 0.0


# ---------------------------------------------------------------------------
# TestRecommendedMarker
# ---------------------------------------------------------------------------
class TestRecommendedMarker:
    """Recommended marker = lowest E[cost]."""

    def test_recommended_is_min_cost(self, engine, process_anomaly_medium):
        """Recommended option must have minimum expected cost."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        recommended = [o for o in envelope.options if o.is_recommended][0]
        costs = [o.expected_cost for o in envelope.options]
        assert recommended.expected_cost == min(costs)

    def test_observation_always_recommended_when_available(self, engine, process_anomaly_medium):
        """OBSERVATION (0 TL) should always be recommended when available."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        obs_options = [o for o in envelope.options if o.work_order_type == WorkOrderType.OBSERVATION]
        if obs_options:
            assert obs_options[0].expected_cost == 0.0
            assert obs_options[0].is_recommended is True

    def test_only_one_recommended(self, engine, process_anomaly_medium):
        """Only ONE option can be recommended."""
        envelope = engine.generate(diagnosis=process_anomaly_medium)
        recommended_count = sum(1 for o in envelope.options if o.is_recommended)
        assert recommended_count == 1


# ---------------------------------------------------------------------------
# TestAntiPattern
# ---------------------------------------------------------------------------
class TestAntiPattern:
    """Anti-pattern: 3+ same option cannot be recommended (CONTROLLED_STOP exempt)."""

    def test_no_three_same_options(self, engine):
        """Cannot have 3+ options of the same WorkOrderType."""
        # This tests the engine's internal guard
        diagnosis = {
            "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
            "machine_id": "AC-001",
            "confidence": 0.8,
            "mode_id": "bearing_outer_race",
            "rpn": 150,
            "severity": Severity.MEDIUM,
        }
        envelope = engine.generate(diagnosis=diagnosis)
        type_counts = {}
        for o in envelope.options:
            t = o.work_order_type
            type_counts[t] = type_counts.get(t, 0) + 1

        for wtype, count in type_counts.items():
            if wtype != WorkOrderType.CONTROLLED_STOP:
                assert count < ANTI_PATTERN_MAX_REPEATS, \
                    f"Anti-pattern: {wtype} appears {count} times (max {ANTI_PATTERN_MAX_REPEATS - 1})"

    def test_controlled_stop_exempt_from_anti_pattern(self, engine):
        """CONTROLLED_STOP is exempt from the anti-pattern rule."""
        # CONTROLLED_STOP can appear multiple times
        assert CONTROLLED_STOP_EXEMPT is True

    def test_anti_pattern_guard_raises_on_violation(self, engine):
        """Engine should raise or reject if anti-pattern would be violated."""
        # Attempting to add 3+ same type should be rejected
        with pytest.raises((ValueError, RuntimeError)):
            engine._validate_options([
                DecisionOption(work_order_type=WorkOrderType.SLOWDOWN_ORDER, expected_cost=100.0),
                DecisionOption(work_order_type=WorkOrderType.SLOWDOWN_ORDER, expected_cost=200.0),
                DecisionOption(work_order_type=WorkOrderType.SLOWDOWN_ORDER, expected_cost=300.0),
            ])


# ---------------------------------------------------------------------------
# TestWorkOrderTypes
# ---------------------------------------------------------------------------
class TestWorkOrderTypes:
    """WorkOrderType enum: OBSERVATION, TECHNICAL_DISPATCH, SLOWDOWN_ORDER,
    STOP_PREP_ORDER, CONTROLLED_STOP."""

    def test_observation_type_exists(self):
        """OBSERVATION type must exist."""
        assert WorkOrderType.OBSERVATION is not None

    def test_technical_dispatch_type_exists(self):
        """TECHNICAL_DISPATCH type must exist."""
        assert WorkOrderType.TECHNICAL_DISPATCH is not None

    def test_slowdown_order_type_exists(self):
        """SLOWDOWN_ORDER type must exist."""
        assert WorkOrderType.SLOWDOWN_ORDER is not None

    def test_stop_prep_order_type_exists(self):
        """STOP_PREP_ORDER type must exist."""
        assert WorkOrderType.STOP_PREP_ORDER is not None

    def test_controlled_stop_type_exists(self):
        """CONTROLLED_STOP type must exist."""
        assert WorkOrderType.CONTROLLED_STOP is not None

    def test_all_five_types_defined(self):
        """All 5 WorkOrderTypes must be defined."""
        assert len(WorkOrderType) == 5


# ---------------------------------------------------------------------------
# TestRPNBoundaries
# ---------------------------------------------------------------------------
class TestRPNBoundaries:
    """RPN boundary tests for severity classification."""

    def test_rpn_exactly_200_is_medium(self, engine):
        """RPN = 200 should be medium severity (boundary)."""
        diagnosis = {
            "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
            "machine_id": "AC-001",
            "confidence": 0.8,
            "mode_id": "bearing",
            "rpn": 200,
            "severity": Severity.MEDIUM,
        }
        envelope = engine.generate(diagnosis=diagnosis)
        assert len(envelope.options) == 3  # medium → 3 options

    def test_rpn_201_is_high(self, engine):
        """RPN = 201 should be high severity."""
        diagnosis = {
            "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
            "machine_id": "AC-001",
            "confidence": 0.8,
            "mode_id": "bearing",
            "rpn": 201,
            "severity": Severity.HIGH,
        }
        envelope = engine.generate(diagnosis=diagnosis)
        assert len(envelope.options) == 1  # high → 1 option

    def test_rpn_exactly_100_is_medium(self, engine):
        """RPN = 100 should be medium severity (boundary)."""
        diagnosis = {
            "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
            "machine_id": "AC-001",
            "confidence": 0.7,
            "mode_id": "bearing",
            "rpn": 100,
            "severity": Severity.MEDIUM,
        }
        envelope = engine.generate(diagnosis=diagnosis)
        assert len(envelope.options) == 3  # medium → 3 options

    def test_rpn_99_is_low(self, engine):
        """RPN = 99 should be low severity."""
        diagnosis = {
            "diagnosis_type": DiagnosisType.PROCESS_ANOMALY,
            "machine_id": "AC-001",
            "confidence": 0.6,
            "mode_id": "bearing",
            "rpn": 99,
            "severity": Severity.LOW,
        }
        envelope = engine.generate(diagnosis=diagnosis)
        assert len(envelope.options) == 2  # low → 2 options


# ---------------------------------------------------------------------------
# TestDecisionOptionStructure
# ---------------------------------------------------------------------------
class TestDecisionOptionStructure:
    """DecisionOption dataclass validation."""

    def test_option_has_work_order_type(self):
        """Option must have a work_order_type."""
        opt = DecisionOption(
            work_order_type=WorkOrderType.OBSERVATION,
            expected_cost=0.0,
        )
        assert opt.work_order_type == WorkOrderType.OBSERVATION

    def test_option_has_expected_cost(self):
        """Option must have expected_cost."""
        opt = DecisionOption(
            work_order_type=WorkOrderType.SLOWDOWN_ORDER,
            expected_cost=1500.0,
        )
        assert opt.expected_cost == 1500.0

    def test_option_has_is_recommended_default_false(self):
        """is_recommended defaults to False."""
        opt = DecisionOption(
            work_order_type=WorkOrderType.OBSERVATION,
            expected_cost=0.0,
        )
        assert opt.is_recommended is False

    def test_observation_cost_is_zero(self):
        """OBSERVATION option must have 0 cost."""
        opt = DecisionOption(
            work_order_type=WorkOrderType.OBSERVATION,
            expected_cost=0.0,
        )
        assert opt.expected_cost == 0.0
