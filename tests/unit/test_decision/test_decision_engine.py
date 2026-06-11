"""
Unit tests for src.decision.decision_engine (Phase 2C - Decision Engine Layer)

Tests DecisionEngine: 4-scenario decision framework
  - OBSERVE = 0 TL (Murat's rule #1)
  - REDUCE_LOAD: effective only when Weibull β > 1, rejected for Stage IV bearing
  - PLANNED: cost-optimal block replacement model
  - SHUTDOWN: safe emergency stop, excluded when P(survive) < 0.40

Also tests:
  - Cascade cost propagation (AC → HX + CM idle)
  - Machine physics profiles (AC, HX, CM)
  - Survival probability: Lognormal(CV=0.25)
  - Prescriptive action map

NOTE: These tests will FAIL until decision_engine.py is implemented.
"""


import pytest

# ---------------------------------------------------------------------------
# Import targets - will exist after coder agent migration
# ---------------------------------------------------------------------------
from src.decision.decision_engine import (
    SHUTDOWN_SURVIVAL_THRESHOLD,
    SURVIVAL_CV,
    DecisionEngine,
    DecisionScenario,
    MachineProfile,
    SurvivalModel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# These should be defined in decision_engine.py
# WEIBULL_EFFECTIVE_THRESHOLD = 1.0  (β > 1 means wear-out dominant)
# SHUTDOWN_SURVIVAL_THRESHOLD = 0.40  (exclude if P(survive) < 0.40)
# SURVIVAL_CV = 0.25  (Lognormal coefficient of variation)
# STAGE_IV_BEARING_REJECTION = True  (REDUCE_LOAD rejected for Stage IV)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def engine():
    """Fresh DecisionEngine with default config."""
    return DecisionEngine()


@pytest.fixture
def ac_profile():
    """Air Compressor machine profile."""
    return MachineProfile(
        machine_id="AC-001",
        machine_type="air_compressor",
        production_rate_per_hour=8000.0,
        cascade_targets=["HX-001", "CM-001"],
        weibull_beta=2.5,
        weibull_eta=5000.0,
        bearing_stage="II",
    )


@pytest.fixture
def hx_profile():
    """Heat Exchanger machine profile."""
    return MachineProfile(
        machine_id="HX-001",
        machine_type="heat_exchanger",
        production_rate_per_hour=6000.0,
        cascade_targets=["CM-001"],
        weibull_beta=1.8,
        weibull_eta=8000.0,
        bearing_stage="I",
    )


@pytest.fixture
def cm_profile():
    """Cooling Machine profile."""
    return MachineProfile(
        machine_id="CM-001",
        machine_type="cooling_machine",
        production_rate_per_hour=3000.0,
        cascade_targets=[],
        weibull_beta=3.0,
        weibull_eta=4000.0,
        bearing_stage="III",
    )


@pytest.fixture
def stage_iv_bearing_profile():
    """Machine with Stage IV bearing damage (REDUCE_LOAD should be rejected)."""
    return MachineProfile(
        machine_id="AC-002",
        machine_type="air_compressor",
        production_rate_per_hour=7000.0,
        cascade_targets=["HX-001"],
        weibull_beta=2.0,
        weibull_eta=3000.0,
        bearing_stage="IV",
    )


@pytest.fixture
def low_weibull_profile():
    """Machine with Weibull β < 1 (random failures, REDUCE_LOAD ineffective)."""
    return MachineProfile(
        machine_id="AC-003",
        machine_type="air_compressor",
        production_rate_per_hour=5000.0,
        cascade_targets=[],
        weibull_beta=0.7,
        weibull_eta=6000.0,
        bearing_stage="I",
    )


# ---------------------------------------------------------------------------
# TestOBSERVEZeroCost - KRİTİK: Murat's Rule #1
# ---------------------------------------------------------------------------
class TestOBSERVEZeroCost:
    """
    KRİTİK: OBSERVE scenario must ALWAYS be 0 TL.
    This is Murat's #1 rule - observation costs nothing.
    """

    def test_observe_cost_is_zero(self, engine, ac_profile):
        """OBSERVE scenario must return 0 TL cost."""
        result = engine.evaluate(
            scenario=DecisionScenario.OBSERVE,
            machine_profile=ac_profile,
        )
        assert result.cost == 0.0

    def test_observe_cost_never_positive_any_rul(self, engine, ac_profile):
        """No matter the RUL, OBSERVE is always 0 TL."""
        for rul_hours in [0.5, 1.0, 4.0, 24.0, 100.0, 1000.0]:
            result = engine.evaluate(
                scenario=DecisionScenario.OBSERVE,
                machine_profile=ac_profile,
                rul_hours=rul_hours,
            )
            assert result.cost == 0.0, f"OBSERVE should be 0 for RUL={rul_hours}"

    def test_observe_cost_zero_regardless_of_bearing_stage(self, engine):
        """OBSERVE = 0 TL even for Stage IV bearing."""
        for stage in ["I", "II", "III", "IV"]:
            profile = MachineProfile(
                machine_id="TEST-001",
                machine_type="air_compressor",
                production_rate_per_hour=5000.0,
                cascade_targets=[],
                weibull_beta=2.0,
                weibull_eta=5000.0,
                bearing_stage=stage,
            )
            result = engine.evaluate(
                scenario=DecisionScenario.OBSERVE,
                machine_profile=profile,
            )
            assert result.cost == 0.0, f"OBSERVE should be 0 for stage={stage}"

    def test_observe_breakdown_all_zero(self, engine, ac_profile):
        """All cost breakdown components should be 0 for OBSERVE."""
        result = engine.evaluate(
            scenario=DecisionScenario.OBSERVE,
            machine_profile=ac_profile,
        )
        assert result.production_loss == 0.0
        assert result.cascade_cost == 0.0
        assert result.labor_cost == 0.0
        assert result.emergency_cost == 0.0

    def test_observe_is_always_recommended_first(self, engine, ac_profile):
        """OBSERVE should always be available as an option."""
        result = engine.evaluate(
            scenario=DecisionScenario.OBSERVE,
            machine_profile=ac_profile,
        )
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# TestReduceLoadScenario
# ---------------------------------------------------------------------------
class TestReduceLoadScenario:
    """
    REDUCE_LOAD: Effective only when Weibull β > 1 (wear-out dominant).
    Rejected for Stage IV bearing damage.
    """

    def test_reduce_load_effective_when_weibull_beta_gt_1(self, engine, ac_profile):
        """REDUCE_LOAD is valid when β > 1 (wear-out failure mode)."""
        assert ac_profile.weibull_beta > 1.0
        result = engine.evaluate(
            scenario=DecisionScenario.REDUCE_LOAD,
            machine_profile=ac_profile,
        )
        assert result.is_valid is True

    def test_reduce_load_ineffective_when_weibull_beta_lt_1(self, engine, low_weibull_profile):
        """REDUCE_LOAD is INVALID when β < 1 (random failures)."""
        assert low_weibull_profile.weibull_beta < 1.0
        result = engine.evaluate(
            scenario=DecisionScenario.REDUCE_LOAD,
            machine_profile=low_weibull_profile,
        )
        assert result.is_valid is False

    def test_reduce_load_ineffective_when_weibull_beta_eq_1(self, engine):
        """REDUCE_LOAD is INVALID when β = 1 (constant failure rate)."""
        profile = MachineProfile(
            machine_id="TEST-001",
            machine_type="pump",
            production_rate_per_hour=4000.0,
            cascade_targets=[],
            weibull_beta=1.0,
            weibull_eta=5000.0,
            bearing_stage="II",
        )
        result = engine.evaluate(
            scenario=DecisionScenario.REDUCE_LOAD,
            machine_profile=profile,
        )
        assert result.is_valid is False

    def test_reduce_load_rejected_stage_iv_bearing(self, engine, stage_iv_bearing_profile):
        """KRİTİK: REDUCE_LOAD must be REJECTED for Stage IV bearing."""
        assert stage_iv_bearing_profile.bearing_stage == "IV"
        result = engine.evaluate(
            scenario=DecisionScenario.REDUCE_LOAD,
            machine_profile=stage_iv_bearing_profile,
        )
        assert result.is_valid is False
        assert "Stage IV" in result.rejection_reason or "stage_iv" in result.rejection_reason.lower()

    def test_reduce_load_accepted_stage_i_ii_iii(self, engine):
        """REDUCE_LOAD is valid for bearing stages I, II, III."""
        for stage in ["I", "II", "III"]:
            profile = MachineProfile(
                machine_id=f"TEST-{stage}",
                machine_type="air_compressor",
                production_rate_per_hour=5000.0,
                cascade_targets=[],
                weibull_beta=2.0,
                weibull_eta=5000.0,
                bearing_stage=stage,
            )
            result = engine.evaluate(
                scenario=DecisionScenario.REDUCE_LOAD,
                machine_profile=profile,
            )
            assert result.is_valid is True, f"REDUCE_LOAD should be valid for stage={stage}"

    def test_reduce_load_reduces_wear_rate(self, engine, ac_profile):
        """REDUCE_LOAD should show reduced wear in the result."""
        result = engine.evaluate(
            scenario=DecisionScenario.REDUCE_LOAD,
            machine_profile=ac_profile,
            load_reduction_percent=20.0,
        )
        assert result.wear_reduction_factor > 0.0
        assert result.wear_reduction_factor < 1.0

    def test_reduce_load_cost_is_production_loss(self, engine, ac_profile):
        """REDUCE_LOAD cost = partial production loss (not full stop)."""
        result = engine.evaluate(
            scenario=DecisionScenario.REDUCE_LOAD,
            machine_profile=ac_profile,
            load_reduction_percent=20.0,
        )
        assert result.cost > 0.0
        assert result.cost < ac_profile.production_rate_per_hour

    def test_reduce_load_higher_beta_more_effective(self, engine):
        """Higher Weibull β means REDUCE_LOAD is more effective."""
        profiles = []
        for beta in [1.5, 2.5, 4.0]:
            p = MachineProfile(
                machine_id=f"TEST-B{beta}",
                machine_type="pump",
                production_rate_per_hour=5000.0,
                cascade_targets=[],
                weibull_beta=beta,
                weibull_eta=5000.0,
                bearing_stage="II",
            )
            profiles.append(p)

        results = []
        for p in profiles:
            r = engine.evaluate(
                scenario=DecisionScenario.REDUCE_LOAD,
                machine_profile=p,
                load_reduction_percent=20.0,
            )
            results.append(r)

        # Higher β → more wear reduction benefit
        assert results[0].wear_reduction_factor >= results[1].wear_reduction_factor
        assert results[1].wear_reduction_factor >= results[2].wear_reduction_factor


# ---------------------------------------------------------------------------
# TestPlannedScenario
# ---------------------------------------------------------------------------
class TestPlannedScenario:
    """
    PLANNED: Cost-optimal block replacement model.
    Minimizes E[cost] = C_plan * (1/RUL) + C_unplanned * (1 - R(t))
    """

    def test_planned_has_positive_cost(self, engine, ac_profile):
        """PLANNED replacement has a positive (but optimized) cost."""
        result = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=200.0,
        )
        assert result.cost > 0.0

    def test_planned_cost_includes_parts_and_labor(self, engine, ac_profile):
        """PLANNED cost should include parts + labor components."""
        result = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=200.0,
        )
        assert result.labor_cost > 0.0
        assert result.parts_cost > 0.0

    def test_planned_cheaper_than_emergency(self, engine, ac_profile):
        """PLANNED should be cheaper than unplanned emergency replacement."""
        planned = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=200.0,
        )
        emergency = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=2.0,
        )
        assert planned.cost < emergency.cost

    def test_planned_optimal_replacement_time(self, engine, ac_profile):
        """PLANNED should find cost-optimal replacement window."""
        result = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=500.0,
        )
        assert result.optimal_replacement_hours is not None
        assert result.optimal_replacement_hours > 0

    def test_planned_longer_rul_lower_urgency(self, engine, ac_profile):
        """Longer RUL should result in lower cost rate."""
        short_rul = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=50.0,
        )
        long_rul = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=500.0,
        )
        # Longer RUL → more time to plan → lower cost pressure
        assert long_rul.cost_rate <= short_rul.cost_rate

    def test_planned_no_cascade_cost(self, engine, ac_profile):
        """PLANNED replacement should have no cascade risk (scheduled)."""
        result = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=200.0,
        )
        assert result.cascade_cost == 0.0


# ---------------------------------------------------------------------------
# TestShutdownScenario
# ---------------------------------------------------------------------------
class TestShutdownScenario:
    """
    SHUTDOWN: Safe emergency stop.
    Excluded when P(survive) < 0.40 (machine won't survive long enough).
    """

    def test_shutdown_has_positive_cost(self, engine, ac_profile):
        """SHUTDOWN has a positive cost (production loss + emergency)."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=2.0,
        )
        assert result.cost > 0.0

    def test_shutdown_excluded_when_survival_below_threshold(self, engine, ac_profile):
        """KRİTİK: SHUTDOWN excluded when P(survive) < 0.40."""
        # Very low RUL → low survival probability
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=0.5,  # 30 minutes - likely won't survive
        )
        assert result.is_valid is False
        assert result.survival_probability < SHUTDOWN_SURVIVAL_THRESHOLD

    def test_shutdown_included_when_survival_above_threshold(self, engine, ac_profile):
        """SHUTDOWN valid when P(survive) >= 0.40."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=48.0,  # 2 days - should survive
        )
        assert result.is_valid is True
        assert result.survival_probability >= SHUTDOWN_SURVIVAL_THRESHOLD

    def test_shutdown_survival_probability_uses_lognormal(self, engine, ac_profile):
        """Survival probability should use Lognormal distribution with CV=0.25."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=24.0,
        )
        assert 0.0 <= result.survival_probability <= 1.0

    def test_shutdown_includes_emergency_cost(self, engine, ac_profile):
        """SHUTDOWN includes emergency response cost."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=4.0,
        )
        assert result.emergency_cost > 0.0

    def test_shutdown_includes_production_loss(self, engine, ac_profile):
        """SHUTDOWN includes full production loss."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=4.0,
        )
        assert result.production_loss > 0.0

    def test_shutdown_boundary_survival_exactly_threshold(self, engine, ac_profile):
        """Edge case: P(survive) exactly at threshold (0.40)."""
        # This tests the boundary condition
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=12.0,  # boundary case
        )
        # Should be valid if >= 0.40, invalid if < 0.40
        if result.survival_probability >= SHUTDOWN_SURVIVAL_THRESHOLD:
            assert result.is_valid is True
        else:
            assert result.is_valid is False


# ---------------------------------------------------------------------------
# TestCascadeCost
# ---------------------------------------------------------------------------
class TestCascadeCost:
    """
    Cascade cost propagation: AC → HX + CM idle.
    When a machine stops, downstream machines also stop.
    """

    def test_cascade_cost_includes_downstream(self, engine, ac_profile):
        """AC shutdown should cascade to HX and CM."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=2.0,
        )
        assert result.cascade_cost > 0.0

    def test_cascade_cost_proportional_to_downstream_production(self, engine):
        """Cascade cost should reflect downstream production rates."""
        # AC has 2 cascade targets (HX + CM)
        ac_profile = MachineProfile(
            machine_id="AC-001",
            machine_type="air_compressor",
            production_rate_per_hour=8000.0,
            cascade_targets=["HX-001", "CM-001"],
            weibull_beta=2.5,
            weibull_eta=5000.0,
            bearing_stage="II",
        )
        # Machine with no cascade targets
        isolated_profile = MachineProfile(
            machine_id="ISO-001",
            machine_type="standalone",
            production_rate_per_hour=8000.0,
            cascade_targets=[],
            weibull_beta=2.5,
            weibull_eta=5000.0,
            bearing_stage="II",
        )

        ac_result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=4.0,
        )
        iso_result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=isolated_profile,
            rul_hours=4.0,
        )
        # AC should have higher cascade cost (has downstream machines)
        assert ac_result.cascade_cost > iso_result.cascade_cost

    def test_no_cascade_for_leaf_machine(self, engine, cm_profile):
        """CM-001 (leaf node) should have zero cascade cost."""
        assert cm_profile.cascade_targets == []
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=cm_profile,
            rul_hours=4.0,
        )
        assert result.cascade_cost == 0.0


# ---------------------------------------------------------------------------
# TestMachinePhysicsProfiles
# ---------------------------------------------------------------------------
class TestMachinePhysicsProfiles:
    """Machine-specific physics profiles (AC, HX, CM)."""

    def test_ac_profile_weibull_parameters(self, ac_profile):
        """AC should have wear-out dominant Weibull (β > 1)."""
        assert ac_profile.weibull_beta > 1.0
        assert ac_profile.weibull_eta > 0

    def test_hx_profile_weibull_parameters(self, hx_profile):
        """HX should have wear-out dominant Weibull."""
        assert hx_profile.weibull_beta > 1.0

    def test_cm_profile_weibull_parameters(self, cm_profile):
        """CM should have wear-out dominant Weibull."""
        assert cm_profile.weibull_beta > 1.0

    def test_machine_profile_cascade_chain(self, ac_profile):
        """AC → HX → CM cascade chain."""
        assert "HX-001" in ac_profile.cascade_targets

    def test_different_machines_different_costs(self, engine):
        """Different machine types should produce different costs."""
        ac = MachineProfile(
            machine_id="AC-001", machine_type="air_compressor",
            production_rate_per_hour=8000.0, cascade_targets=[],
            weibull_beta=2.5, weibull_eta=5000.0, bearing_stage="II",
        )
        cm = MachineProfile(
            machine_id="CM-001", machine_type="cooling_machine",
            production_rate_per_hour=3000.0, cascade_targets=[],
            weibull_beta=3.0, weibull_eta=4000.0, bearing_stage="III",
        )
        ac_result = engine.evaluate(scenario=DecisionScenario.PLANNED, machine_profile=ac, rul_hours=200.0)
        cm_result = engine.evaluate(scenario=DecisionScenario.PLANNED, machine_profile=cm, rul_hours=200.0)
        assert ac_result.cost != cm_result.cost


# ---------------------------------------------------------------------------
# TestSurvivalProbability
# ---------------------------------------------------------------------------
class TestSurvivalProbability:
    """Survival probability: Lognormal(CV=0.25)."""

    def test_survival_model_lognormal_cv(self, engine):
        """Survival model should use Lognormal with CV=0.25."""
        model = SurvivalModel(cv=SURVIVAL_CV)
        assert model.cv == SURVIVAL_CV

    def test_survival_probability_decreases_with_lower_rul(self, engine, ac_profile):
        """Lower RUL → lower survival probability."""
        high_rul = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=100.0,
        )
        low_rul = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=1.0,
        )
        assert high_rul.survival_probability > low_rul.survival_probability

    def test_survival_probability_bounded(self, engine, ac_profile):
        """Survival probability must be in [0, 1]."""
        for rul in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            result = engine.evaluate(
                scenario=DecisionScenario.SHUTDOWN,
                machine_profile=ac_profile,
                rul_hours=rul,
            )
            assert 0.0 <= result.survival_probability <= 1.0

    def test_survival_probability_very_high_rul(self, engine, ac_profile):
        """Very high RUL → survival probability approaches 1.0."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=10000.0,
        )
        assert result.survival_probability > 0.95


# ---------------------------------------------------------------------------
# TestPrescriptiveActionMap
# ---------------------------------------------------------------------------
class TestPrescriptiveActionMap:
    """Prescriptive action map: scenario → specific actions."""

    def test_observe_action_is_monitor(self, engine, ac_profile):
        """OBSERVE → monitor action (no physical intervention)."""
        result = engine.evaluate(
            scenario=DecisionScenario.OBSERVE,
            machine_profile=ac_profile,
        )
        assert result.prescriptive_action is not None
        assert result.prescriptive_action.action_type == "MONITOR"

    def test_reduce_load_action_is_slowdown(self, engine, ac_profile):
        """REDUCE_LOAD → slowdown action."""
        result = engine.evaluate(
            scenario=DecisionScenario.REDUCE_LOAD,
            machine_profile=ac_profile,
            load_reduction_percent=20.0,
        )
        assert result.prescriptive_action is not None
        assert result.prescriptive_action.action_type == "SLOWDOWN"

    def test_planned_action_is_schedule_replacement(self, engine, ac_profile):
        """PLANNED → schedule replacement action."""
        result = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=200.0,
        )
        assert result.prescriptive_action is not None
        assert result.prescriptive_action.action_type == "SCHEDULE_REPLACEMENT"

    def test_shutdown_action_is_emergency_stop(self, engine, ac_profile):
        """SHUTDOWN → emergency stop action."""
        result = engine.evaluate(
            scenario=DecisionScenario.SHUTDOWN,
            machine_profile=ac_profile,
            rul_hours=4.0,
        )
        if result.is_valid:
            assert result.prescriptive_action.action_type == "EMERGENCY_STOP"


# ---------------------------------------------------------------------------
# TestDecisionEngineIntegration
# ---------------------------------------------------------------------------
class TestDecisionEngineIntegration:
    """Integration tests for the full decision pipeline."""

    def test_evaluate_all_scenarios(self, engine, ac_profile):
        """Engine should evaluate all 4 scenarios."""
        scenarios = [
            DecisionScenario.OBSERVE,
            DecisionScenario.REDUCE_LOAD,
            DecisionScenario.PLANNED,
            DecisionScenario.SHUTDOWN,
        ]
        results = []
        for s in scenarios:
            r = engine.evaluate(
                scenario=s,
                machine_profile=ac_profile,
                rul_hours=24.0,
            )
            results.append(r)
        assert len(results) == 4

    def test_observe_always_cheapest(self, engine, ac_profile):
        """OBSERVE should always be the cheapest option."""
        observe = engine.evaluate(
            scenario=DecisionScenario.OBSERVE,
            machine_profile=ac_profile,
            rul_hours=24.0,
        )
        planned = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=24.0,
        )
        assert observe.cost <= planned.cost

    def test_recommend_options_sorted_by_expected_cost(self, engine, ac_profile):
        """Options are sorted by risk-adjusted expected cost."""
        options = engine.recommend(
            machine_profile=ac_profile,
            rul_hours=24.0,
        )
        if len(options) > 1:
            expected_costs = [o.expected_cost for o in options]
            assert expected_costs == sorted(expected_costs)

    def test_recommended_marker_is_lowest_expected_cost(self, engine, ac_profile):
        """The 'recommended' option has the lowest expected cost."""
        options = engine.recommend(
            machine_profile=ac_profile,
            rul_hours=24.0,
        )
        recommended = [o for o in options if o.is_recommended]
        if recommended:
            min_expected = min(o.expected_cost for o in options)
            assert recommended[0].expected_cost == min_expected

    def test_high_rul_recommends_observe(self, engine, ac_profile):
        """A healthy machine (RUL >> horizon) should just be observed."""
        options = engine.recommend(machine_profile=ac_profile, rul_hours=200.0)
        recommended = next(o for o in options if o.is_recommended)
        assert recommended.scenario == DecisionScenario.OBSERVE

    def test_low_rul_never_recommends_observe(self, engine, ac_profile):
        """A failing machine must trigger an intervention, never OBSERVE."""
        options = engine.recommend(machine_profile=ac_profile, rul_hours=8.0)
        recommended = next(o for o in options if o.is_recommended)
        assert recommended.scenario != DecisionScenario.OBSERVE

    def test_critical_rul_recommends_shutdown(self, engine, ac_profile):
        """With hours left, a controlled stop beats waiting for the crash."""
        options = engine.recommend(machine_profile=ac_profile, rul_hours=2.0)
        recommended = next(o for o in options if o.is_recommended)
        assert recommended.scenario == DecisionScenario.SHUTDOWN

    def test_observe_expected_cost_prices_failure_risk(self, engine, ac_profile):
        """OBSERVE stays free in direct cost but carries the failure risk."""
        result = engine.evaluate(
            scenario=DecisionScenario.OBSERVE,
            machine_profile=ac_profile,
            rul_hours=6.0,
        )
        assert result.cost == 0.0
        assert result.expected_cost > 0.0
        assert 0.0 < result.failure_probability <= 1.0

    def test_run_to_failure_exceeds_planned_repair(self, engine, ac_profile):
        """The reactive counterfactual must cost more than planned work."""
        rtf = engine.run_to_failure_cost(ac_profile)
        planned = engine.evaluate(
            scenario=DecisionScenario.PLANNED,
            machine_profile=ac_profile,
            rul_hours=24.0,
        )
        assert rtf > planned.cost
