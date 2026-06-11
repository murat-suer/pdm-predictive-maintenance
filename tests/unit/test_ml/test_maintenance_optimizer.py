"""Tests for Cost-Aware Maintenance Optimizer."""
from src.ml.maintenance_optimizer import MaintenanceCostOptimizer


class TestMaintenanceCostOptimizer:
    def test_weibull_reliability(self):
        """Reliability should decrease from 1.0 to 0.0."""
        opt = MaintenanceCostOptimizer()

        R_0 = opt.weibull_reliability(0.0, eta=500.0, beta=2.0)
        R_mid = opt.weibull_reliability(250.0, eta=500.0, beta=2.0)
        R_end = opt.weibull_reliability(1000.0, eta=500.0, beta=2.0)

        assert R_0 == 1.0, "R(0) should be 1.0"
        assert 0.0 < R_mid < 1.0, "R(t) should be between 0 and 1"
        assert R_0 > R_mid > R_end, "R(t) should decrease with time"

    def test_find_optimal_replacement_time(self):
        """Should find tp* that minimizes cost rate."""
        opt = MaintenanceCostOptimizer(
            preventive_cost=1000.0,
            corrective_cost=5000.0
        )

        tp_opt, cost_rate = opt.find_optimal_replacement_time(eta=500.0, beta=2.0)

        assert tp_opt > 0, "Optimal time should be positive"
        assert cost_rate > 0, "Cost rate should be positive"

        # Optimal should be less than mean life (preventive before failure)
        import math
        mttf = 500.0 * math.gamma(1 + 1/2.0)
        assert tp_opt < mttf, "Preventive should occur before mean failure time"

    def test_preventive_better_than_corrective(self):
        """Preventive maintenance should have lower cost rate than corrective."""
        opt = MaintenanceCostOptimizer(
            preventive_cost=1000.0,
            corrective_cost=5000.0
        )

        tp_opt, _ = opt.find_optimal_replacement_time(eta=500.0, beta=2.0)
        comparison = opt.compare_strategies(eta=500.0, beta=2.0, tp_optimal=tp_opt)

        assert comparison['preventive_cost_rate'] < comparison['corrective_cost_rate']
        assert comparison['savings_percent'] > 0, "Should have positive savings"

    def test_high_corrective_cost_drives_earlier_prevention(self):
        """Higher corrective cost should lead to earlier preventive maintenance."""
        opt_low = MaintenanceCostOptimizer(preventive_cost=1000.0, corrective_cost=2000.0)
        opt_high = MaintenanceCostOptimizer(preventive_cost=1000.0, corrective_cost=10000.0)

        tp_low, _ = opt_low.find_optimal_replacement_time(eta=500.0, beta=2.0)
        tp_high, _ = opt_high.find_optimal_replacement_time(eta=500.0, beta=2.0)

        assert tp_high < tp_low, "Higher corrective cost should drive earlier prevention"
