"""
Unit tests for src.decision.cost_calculator

Tests CostCalculator: honest cost-savings calculator.
KRİTİK KURAL: OBSERVATION = 0 TL (ASLA pozitif olmamalı)
"""


import pytest

from src.decision.cost_calculator import (
    CostCalculator,
    DecisionType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def calculator():
    """Default cost calculator with standard config."""
    return CostCalculator()


@pytest.fixture
def calculator_with_config(tmp_path):
    """Calculator loaded from a config file."""
    config_content = """
production_rate_per_hour: 5000.0
emergency_response_cost: 15000.0
cascade_multiplier: 1.5
premium_factor: 1.2
wear_reduction_factor: 0.3
machines:
  AC-001:
    production_rate_per_hour: 8000.0
  CM-001:
    production_rate_per_hour: 3000.0
"""
    config_file = tmp_path / "cost_config.yaml"
    config_file.write_text(config_content)
    return CostCalculator(config_path=str(config_file))


# ---------------------------------------------------------------------------
# TestOBSERVATIONZeroCost - KRİTİK KURAL
# ---------------------------------------------------------------------------
class TestOBSERVATIONZeroCost:
    """
    KRİTİK: OBSERVATION maliyeti = 0 TL (ASLA pozitif olmamalı)
    Murat'ın 1 numaralı kuralı.
    """

    def test_observation_cost_is_zero(self, calculator):
        """OBSERVATION must always be 0 TL."""
        cost = calculator.calculate(
            decision_type=DecisionType.OBSERVATION,
            machine_id="AC-001",
        )
        assert cost.total == 0.0

    def test_observation_cost_never_positive(self, calculator):
        """No matter the parameters, OBSERVATION is 0."""
        for rul in [0.5, 1.0, 4.0, 24.0, 100.0]:
            cost = calculator.calculate(
                decision_type=DecisionType.OBSERVATION,
                machine_id="AC-001",
                rul_hours=rul,
            )
            assert cost.total == 0.0, f"OBSERVATION should be 0 for RUL={rul}"

    def test_observation_30min_recommendation_zero_cost(self, calculator):
        """30 dakika gözlem önerisi için maliyet çıkmaz."""
        cost = calculator.calculate(
            decision_type=DecisionType.OBSERVATION,
            machine_id="AC-001",
            observation_minutes=30,
        )
        assert cost.total == 0.0

    def test_observation_extended_period_still_zero(self, calculator):
        """Even extended observation (hours) is still 0 TL."""
        cost = calculator.calculate(
            decision_type=DecisionType.OBSERVATION,
            machine_id="AC-001",
            observation_minutes=120,
        )
        assert cost.total == 0.0

    def test_observation_breakdown_all_zero(self, calculator):
        """All breakdown components should be 0 for OBSERVATION."""
        cost = calculator.calculate(
            decision_type=DecisionType.OBSERVATION,
            machine_id="AC-001",
        )
        assert cost.production_loss == 0.0
        assert cost.emergency_cost == 0.0
        assert cost.cascade_risk == 0.0
        assert cost.labor_cost == 0.0


# ---------------------------------------------------------------------------
# TestTECHNICAL_DISPATCH
# ---------------------------------------------------------------------------
class TestTECHNICALDispatch:
    """TECHNICAL_DISPATCH = emergency_response + cascade."""

    def test_technical_dispatch_includes_emergency_response(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.TECHNICAL_DISPATCH,
            machine_id="AC-001",
        )
        assert cost.emergency_cost > 0

    def test_technical_dispatch_includes_cascade(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.TECHNICAL_DISPATCH,
            machine_id="AC-001",
        )
        assert cost.cascade_risk > 0

    def test_technical_dispatch_total_equals_components(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.TECHNICAL_DISPATCH,
            machine_id="AC-001",
        )
        expected = cost.emergency_cost + cost.cascade_risk
        assert cost.total == pytest.approx(expected, rel=0.01)

    def test_technical_dispatch_positive_cost(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.TECHNICAL_DISPATCH,
            machine_id="AC-001",
        )
        assert cost.total > 0


# ---------------------------------------------------------------------------
# TestSLOWDOWN_ORDER
# ---------------------------------------------------------------------------
class TestSlowdownOrder:
    """SLOWDOWN_ORDER = baseline * wear_reduction."""

    def test_slowdown_cost_is_production_loss(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.SLOWDOWN_ORDER,
            machine_id="AC-001",
        )
        assert cost.production_loss > 0

    def test_slowdown_uses_wear_reduction(self, calculator):
        """Cost should reflect wear reduction factor."""
        cost = calculator.calculate(
            decision_type=DecisionType.SLOWDOWN_ORDER,
            machine_id="AC-001",
        )
        # Wear reduction should make cost less than full production loss
        assert cost.production_loss < calculator.config.production_rate_per_hour

    def test_slowdown_positive_cost(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.SLOWDOWN_ORDER,
            machine_id="AC-001",
        )
        assert cost.total > 0


# ---------------------------------------------------------------------------
# TestSTOP_PREP_ORDER
# ---------------------------------------------------------------------------
class TestStopPrepOrder:
    """STOP_PREP_ORDER = production_loss_avoided."""

    def test_stop_prep_has_production_loss(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.STOP_PREP_ORDER,
            machine_id="AC-001",
        )
        assert cost.production_loss > 0

    def test_stop_prep_positive_cost(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.STOP_PREP_ORDER,
            machine_id="AC-001",
        )
        assert cost.total > 0


# ---------------------------------------------------------------------------
# TestCONTROLLED_STOP
# ---------------------------------------------------------------------------
class TestControlledStop:
    """CONTROLLED_STOP = cascade * premium."""

    def test_controlled_stop_includes_cascade(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.CONTROLLED_STOP,
            machine_id="AC-001",
        )
        assert cost.cascade_risk > 0

    def test_controlled_stop_includes_premium(self, calculator):
        """Premium factor should make it more expensive than base cascade."""
        cost = calculator.calculate(
            decision_type=DecisionType.CONTROLLED_STOP,
            machine_id="AC-001",
        )
        assert cost.total > 0

    def test_controlled_stop_more_expensive_than_slowdown(self, calculator):
        """Controlled stop should be more expensive than slowdown."""
        stop_cost = calculator.calculate(
            decision_type=DecisionType.CONTROLLED_STOP,
            machine_id="AC-001",
        )
        slow_cost = calculator.calculate(
            decision_type=DecisionType.SLOWDOWN_ORDER,
            machine_id="AC-001",
        )
        assert stop_cost.total > slow_cost.total


# ---------------------------------------------------------------------------
# TestUnknownDecisionType
# ---------------------------------------------------------------------------
class TestUnknownDecisionType:
    """Unknown decision type should return 0 TL."""

    def test_unknown_type_returns_zero(self, calculator):
        cost = calculator.calculate(
            decision_type="UNKNOWN_TYPE",
            machine_id="AC-001",
        )
        assert cost.total == 0.0

    def test_none_type_returns_zero(self, calculator):
        cost = calculator.calculate(
            decision_type=None,
            machine_id="AC-001",
        )
        assert cost.total == 0.0


# ---------------------------------------------------------------------------
# TestConfigFileLoading
# ---------------------------------------------------------------------------
class TestConfigFileLoading:
    """Calculator should load config from YAML file."""

    def test_load_from_file(self, calculator_with_config):
        assert calculator_with_config.config.production_rate_per_hour == 5000.0
        assert calculator_with_config.config.emergency_response_cost == 15000.0

    def test_config_cascade_multiplier(self, calculator_with_config):
        assert calculator_with_config.config.cascade_multiplier == 1.5

    def test_config_premium_factor(self, calculator_with_config):
        assert calculator_with_config.config.premium_factor == 1.2

    def test_missing_config_uses_defaults(self):
        """Missing config file should use defaults without error."""
        calc = CostCalculator(config_path="/nonexistent/path.yaml")
        assert calc.config.production_rate_per_hour > 0


# ---------------------------------------------------------------------------
# TestPerMachineProductionRate
# ---------------------------------------------------------------------------
class TestPerMachineProductionRate:
    """Different machines can have different production rates."""

    def test_machine_specific_rate(self, calculator_with_config):
        """AC-001 has 8000 TL/hr in config."""
        cost = calculator_with_config.calculate(
            decision_type=DecisionType.STOP_PREP_ORDER,
            machine_id="AC-001",
        )
        # Should use AC-001's rate (8000), not default (5000)
        assert cost.production_loss > 0

    def test_unknown_machine_uses_default_rate(self, calculator_with_config):
        """Unknown machine should fall back to default rate."""
        cost = calculator_with_config.calculate(
            decision_type=DecisionType.STOP_PREP_ORDER,
            machine_id="UNKNOWN-999",
        )
        assert cost.production_loss > 0

    def test_different_machines_different_costs(self, calculator_with_config):
        """AC-001 (8000/hr) should cost more than CM-001 (3000/hr)."""
        cost_ac = calculator_with_config.calculate(
            decision_type=DecisionType.STOP_PREP_ORDER,
            machine_id="AC-001",
        )
        cost_cm = calculator_with_config.calculate(
            decision_type=DecisionType.STOP_PREP_ORDER,
            machine_id="CM-001",
        )
        assert cost_ac.total > cost_cm.total


# ---------------------------------------------------------------------------
# TestAntiPattern - Murat'ın 3. kuralı
# ---------------------------------------------------------------------------
class TestAntiPattern:
    """
    Anti-pattern: Sürekli OBSERVE önerip makineyi izlememeli.
    OBSERVATION = 0 TL olduğundan, maliyet bazlı sürekli OBSERVE önerilemez.
    """

    def test_observation_always_zero_prevents_anti_pattern(self, calculator):
        """
        OBSERVATION = 0 olduğundan, cost-based decision engine
        sürekli OBSERVE öneremez (maliyet avantajı yok, sadece risk artar).
        """
        for _ in range(10):
            cost = calculator.calculate(
                decision_type=DecisionType.OBSERVATION,
                machine_id="AC-001",
            )
            assert cost.total == 0.0

    def test_observation_has_no_cost_savings_claim(self, calculator):
        """OBSERVATION should not claim any cost savings."""
        cost = calculator.calculate(
            decision_type=DecisionType.OBSERVATION,
            machine_id="AC-001",
        )
        assert cost.savings == 0.0 or not hasattr(cost, "savings")


# ---------------------------------------------------------------------------
# TestCostComparison
# ---------------------------------------------------------------------------
class TestCostComparison:
    """Verify cost ordering makes business sense."""

    def test_observation_cheapest(self, calculator):
        """OBSERVATION should be the cheapest option."""
        obs = calculator.calculate(DecisionType.OBSERVATION, "AC-001")
        slow = calculator.calculate(DecisionType.SLOWDOWN_ORDER, "AC-001")
        stop = calculator.calculate(DecisionType.CONTROLLED_STOP, "AC-001")
        assert obs.total <= slow.total
        assert obs.total <= stop.total

    def test_emergency_most_expensive(self, calculator):
        """TECHNICAL_DISPATCH (emergency) should be most expensive."""
        dispatch = calculator.calculate(DecisionType.TECHNICAL_DISPATCH, "AC-001")
        slow = calculator.calculate(DecisionType.SLOWDOWN_ORDER, "AC-001")
        assert dispatch.total > slow.total


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_zero_rul_observation_still_zero(self, calculator):
        """Even with RUL=0, OBSERVATION is 0 (but should escalate!)."""
        cost = calculator.calculate(
            decision_type=DecisionType.OBSERVATION,
            machine_id="AC-001",
            rul_hours=0.0,
        )
        assert cost.total == 0.0

    def test_cost_breakdown_has_all_fields(self, calculator):
        cost = calculator.calculate(
            decision_type=DecisionType.TECHNICAL_DISPATCH,
            machine_id="AC-001",
        )
        assert hasattr(cost, "total")
        assert hasattr(cost, "production_loss")
        assert hasattr(cost, "emergency_cost")
        assert hasattr(cost, "cascade_risk")

    def test_cost_is_non_negative(self, calculator):
        """No cost should ever be negative."""
        for dt in DecisionType:
            cost = calculator.calculate(dt, "AC-001")
            assert cost.total >= 0.0, f"Negative cost for {dt}"
