"""Cost-Aware Maintenance Scheduling Optimizer."""
import math

from scipy.integrate import quad
from scipy.optimize import minimize_scalar


class MaintenanceCostOptimizer:
    """Optimizes maintenance timing based on RUL and cost factors.

    Uses the age-replacement policy model: replace at time tp if the unit
    survives, or at failure if it fails before tp. The optimal tp* minimizes
    the long-run expected cost rate.

    Reference: Wang (2002), "A survey to maintenance policy for multi-unit systems"
    """

    def __init__(self,
                 preventive_cost: float = 1000.0,
                 corrective_cost: float = 5000.0,
                 downtime_cost_per_hour: float = 500.0,
                 production_loss_per_hour: float = 2000.0):
        self.Cp = preventive_cost
        self.Cf = corrective_cost
        self.Cd = downtime_cost_per_hour
        self.Cl = production_loss_per_hour

    def weibull_reliability(self, t: float, eta: float, beta: float) -> float:
        """Weibull reliability function: R(t) = exp(-(t/eta)^beta)"""
        if t <= 0:
            return 1.0
        return math.exp(-(t / eta) ** beta)

    def _expected_cycle_length(self, tp: float, eta: float, beta: float) -> float:
        """Expected cycle length = integral_0^tp R(t) dt.

        This is the mean time between replacements (preventive or corrective).
        """
        result, _ = quad(lambda t: self.weibull_reliability(t, eta, beta), 0, tp)
        return result

    def expected_cost_rate(self, tp: float, eta: float, beta: float,
                          downtime_hours: float = 4.0) -> float:
        """
        Calculate expected cost rate for preventive replacement at time tp.

        Expected cost per cycle = Cp * R(tp) + (Cf + Cd * downtime) * (1 - R(tp))
        Expected cycle length   = integral_0^tp R(t) dt

        Cost rate = Expected cost per cycle / Expected cycle length

        Reference: Wang (2002), "A survey to maintenance policy for multi-unit systems"
        """
        R_tp = self.weibull_reliability(tp, eta, beta)

        # Expected cost per cycle
        # Preventive cost if survives, corrective + downtime if fails
        expected_cost = (self.Cp * R_tp +
                        (self.Cf + self.Cd * downtime_hours) * (1 - R_tp))

        # Expected cycle length (integral of reliability)
        expected_length = self._expected_cycle_length(tp, eta, beta)

        if expected_length <= 0:
            return float('inf')

        # Cost rate (cost per unit time)
        return expected_cost / expected_length

    def find_optimal_replacement_time(self, eta: float, beta: float,
                                     tp_min: float = 1.0,
                                     tp_max: float = None) -> tuple[float, float]:
        """
        Find optimal preventive replacement time tp* that minimizes cost rate.

        Returns: (tp_optimal, min_cost_rate)
        """
        if tp_max is None:
            tp_max = eta * 3  # Search up to 3x scale parameter

        result = minimize_scalar(
            self.expected_cost_rate,
            bounds=(tp_min, tp_max),
            args=(eta, beta),
            method='bounded'
        )

        return result.x, result.fun

    def compare_strategies(self, eta: float, beta: float,
                          tp_optimal: float) -> dict[str, float]:
        """Compare preventive vs corrective maintenance strategies."""

        # Preventive at optimal time
        preventive_cost_rate = self.expected_cost_rate(tp_optimal, eta, beta)

        # Corrective only (replace at failure, no preventive action)
        # Mean time to failure = eta * gamma(1 + 1/beta)
        mttf = eta * math.gamma(1 + 1/beta)
        corrective_cost_rate = (self.Cf + self.Cd * 4.0) / mttf

        # Savings
        savings_per_hour = corrective_cost_rate - preventive_cost_rate
        savings_percent = (savings_per_hour / corrective_cost_rate) * 100

        return {
            'preventive_cost_rate': preventive_cost_rate,
            'corrective_cost_rate': corrective_cost_rate,
            'optimal_tp': tp_optimal,
            'savings_per_hour': savings_per_hour,
            'savings_percent': savings_percent
        }
