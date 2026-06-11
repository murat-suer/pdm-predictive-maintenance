"""
Decision Engine (Phase 2C - Decision Engine Layer).

4-scenario decision framework:
  - OBSERVE: 0 TL (Murat's rule #1)
  - REDUCE_LOAD: effective only when Weibull beta > 1, rejected for Stage IV bearing
  - PLANNED: cost-optimal block replacement model
  - SHUTDOWN: safe emergency stop, excluded when P(survive) < 0.40

Also provides:
  - Cascade cost propagation (AC -> HX + CM idle)
  - Machine physics profiles (AC, HX, CM)
  - Survival probability: Lognormal(CV=0.25)
  - Prescriptive action map
"""

import math
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIBULL_EFFECTIVE_THRESHOLD: float = 1.0  # beta > 1 means wear-out dominant
SHUTDOWN_SURVIVAL_THRESHOLD: float = 0.40  # exclude if P(survive) < 0.40
SURVIVAL_CV: float = 0.25  # Lognormal coefficient of variation
STAGE_IV_BEARING_REJECTION: bool = True  # REDUCE_LOAD rejected for Stage IV

# Default cost parameters
DEFAULT_PLANNED_LABOR_COST: float = 5000.0
DEFAULT_PLANNED_PARTS_COST: float = 10000.0
DEFAULT_PLANNED_DOWNTIME_HOURS: float = 4.0
DEFAULT_EMERGENCY_MULTIPLIER: float = 2.5
DEFAULT_SHUTDOWN_HOURS: float = 4.0
DEFAULT_LOAD_REDUCTION_FACTOR: float = 0.5  # (1 - load_reduction/100) ^ beta
DEFAULT_CASCADE_RATE_PER_HOUR: float = 5000.0  # Default rate for unregistered targets
SURVIVAL_MIN_HOURS: float = 1.0  # Minimum hours needed for safe operation

# Reactive (run-to-failure) counterfactual parameters. A surprise failure is
# more expensive than planned work: the crew is called out after the fact,
# has to diagnose a dead machine and fetch parts, and the sudden line stop
# stresses the neighbouring machines.
# Two days: a realistic planning horizon for arranging a maintenance slot,
# and early enough that intervention is recommended while it can still be
# scheduled (demo machines live ~6 days).
DECISION_HORIZON_HOURS: float = 48.0
EMERGENCY_RESPONSE_HOURS: float = 2.0       # call-out + travel after surprise failure
DIAGNOSIS_HOURS: float = 1.5                # fault finding on a dead machine
PARTS_LOGISTICS_HOURS: float = 1.5          # fetch parts from the warehouse
SHIFT_CHANGE_OVERLAP_HOURS: float = 0.5     # planned work reuses the no-production window
COLLATERAL_STRESS_FACTOR: float = 0.10      # share of partners' repair budget consumed by a sudden stop
PLANNED_LEAD_HOURS: float = 8.0             # planned slot lands at the next shift boundary


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class DecisionScenario(str, Enum):
    """The 4 decision scenarios."""
    OBSERVE = "OBSERVE"
    REDUCE_LOAD = "REDUCE_LOAD"
    PLANNED = "PLANNED"
    SHUTDOWN = "SHUTDOWN"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MachineProfile:
    """Machine-specific physics profile."""
    machine_id: str
    machine_type: str
    production_rate_per_hour: float
    cascade_targets: list[str]
    weibull_beta: float
    weibull_eta: float
    bearing_stage: str  # "I", "II", "III", "IV"


@dataclass
class PrescriptiveAction:
    """A prescriptive action recommendation."""
    action_type: str  # MONITOR, SLOWDOWN, SCHEDULE_REPLACEMENT, EMERGENCY_STOP
    description: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass
class DecisionResult:
    """Result of evaluating a decision scenario."""
    scenario: DecisionScenario
    cost: float
    is_valid: bool = True
    rejection_reason: str = ""
    production_loss: float = 0.0
    cascade_cost: float = 0.0
    labor_cost: float = 0.0
    emergency_cost: float = 0.0
    parts_cost: float = 0.0
    survival_probability: float = 1.0
    wear_reduction_factor: float = 1.0
    optimal_replacement_hours: float | None = None
    cost_rate: float = 0.0
    prescriptive_action: PrescriptiveAction | None = None
    # Risk-adjusted expected cost over the decision horizon. Direct cost says
    # what the action itself costs; expected cost additionally prices the
    # failure risk the action leaves open (P(fail) × run-to-failure cost).
    expected_cost: float = 0.0
    failure_probability: float = 0.0


@dataclass
class RecommendedOption:
    """A recommended option from the engine."""
    scenario: DecisionScenario
    cost: float
    is_recommended: bool = False
    is_valid: bool = True
    expected_cost: float = 0.0
    failure_probability: float = 0.0
    # REDUCE_LOAD only: the minimal reduction that bridges the machine
    # safely to the planned repair slot, and the survival probability it
    # buys — the operator sees "how much do I slow down, and what for".
    load_reduction_percent: float | None = None
    survival_to_repair: float | None = None


# ---------------------------------------------------------------------------
# Survival Model
# ---------------------------------------------------------------------------
class SurvivalModel:
    """
    Survival probability using Lognormal distribution.

    P(survive | RUL) = Phi((ln(RUL) - ln(min_hours)) / sigma)
    where sigma is derived from CV=0.25.

    Higher RUL → higher survival probability.
    """

    def __init__(self, cv: float = SURVIVAL_CV):
        self.cv = cv
        # sigma derived from CV: sigma = sqrt(ln(1 + CV^2))
        self.sigma = math.sqrt(math.log(1.0 + cv * cv))

    def survival_probability(self, rul_hours: float, eta: float) -> float:
        """
        Calculate P(survive | RUL hours, characteristic life eta).

        Uses lognormal CDF: P(survive) = Phi((ln(RUL) - ln(min_hours)) / sigma)
        where min_hours is the minimum time needed for safe operation.

        Higher RUL → higher P(survive).
        """
        if rul_hours <= 0:
            return 0.0
        if eta <= 0:
            return 0.0

        # P(survive) = Phi((ln(RUL) - ln(SURVIVAL_MIN_HOURS)) / sigma)
        z = (math.log(rul_hours) - math.log(SURVIVAL_MIN_HOURS)) / self.sigma
        cdf = self._normal_cdf(z)
        return max(0.0, min(1.0, cdf))

    def failure_within(self, rul_hours: float, horizon_hours: float) -> float:
        """P(failure within the horizon | RUL estimate).

        Time-to-failure is modelled Lognormal with median = RUL and the
        model's CV: P(T <= H) = Phi((ln H - ln RUL) / sigma).
        """
        if rul_hours <= 0:
            return 1.0
        if horizon_hours <= 0:
            return 0.0
        z = (math.log(horizon_hours) - math.log(rul_hours)) / self.sigma
        return max(0.0, min(1.0, self._normal_cdf(z)))

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximate standard normal CDF using error function."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Cascade Model
# ---------------------------------------------------------------------------
class CascadeModel:
    """
    Cascade cost propagation model.
    When a machine stops, downstream machines also stop.
    """

    def __init__(self, machine_profiles: dict[str, MachineProfile] | None = None):
        self.machine_profiles = machine_profiles or {}

    def calculate_cascade_cost(
        self,
        machine_profile: MachineProfile,
        downtime_hours: float,
        downstream_profiles: dict[str, MachineProfile] | None = None,
    ) -> float:
        """
        Calculate cascade cost from downstream machines being idle.

        For unregistered cascade targets, uses DEFAULT_CASCADE_RATE_PER_HOUR.

        Args:
            machine_profile: The machine that stopped
            downtime_hours: Hours of downtime
            downstream_profiles: Optional dict of machine_id -> MachineProfile
        """
        profiles = downstream_profiles or self.machine_profiles
        total_cascade = 0.0

        for target_id in machine_profile.cascade_targets:
            if target_id in profiles:
                target = profiles[target_id]
                total_cascade += target.production_rate_per_hour * downtime_hours
            else:
                # Use default rate for unregistered targets
                total_cascade += DEFAULT_CASCADE_RATE_PER_HOUR * downtime_hours

        return total_cascade


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    Main decision engine implementing 4-scenario framework.

    Evaluates OBSERVE, REDUCE_LOAD, PLANNED, and SHUTDOWN scenarios
    with machine physics, survival probability, and cascade costs.
    """

    def __init__(
        self,
        survival_model: SurvivalModel | None = None,
        cascade_model: CascadeModel | None = None,
        planned_labor_cost: float = DEFAULT_PLANNED_LABOR_COST,
        planned_parts_cost: float = DEFAULT_PLANNED_PARTS_COST,
        emergency_multiplier: float = DEFAULT_EMERGENCY_MULTIPLIER,
        shutdown_hours: float = DEFAULT_SHUTDOWN_HOURS,
    ):
        self.survival_model = survival_model or SurvivalModel(cv=SURVIVAL_CV)
        self.cascade_model = cascade_model or CascadeModel()
        self.planned_labor_cost = planned_labor_cost
        self.planned_parts_cost = planned_parts_cost
        self.emergency_multiplier = emergency_multiplier
        self.shutdown_hours = shutdown_hours

    def run_to_failure_cost(self, machine_profile: MachineProfile) -> float:
        """Counterfactual: what a surprise run-to-failure event costs.

        Compared to planned work the reactive case pays the emergency
        premium on repair, a much longer downtime (call-out + diagnosis +
        parts logistics on top of the repair itself), the cascade loss of
        the stopped line, and collateral stress on neighbouring machines
        from the sudden stop.
        """
        repair_cost = (
            (self.planned_labor_cost + self.planned_parts_cost)
            * self.emergency_multiplier
        )
        downtime_hours = (
            DEFAULT_PLANNED_DOWNTIME_HOURS
            + EMERGENCY_RESPONSE_HOURS
            + DIAGNOSIS_HOURS
            + PARTS_LOGISTICS_HOURS
        )
        production_loss = machine_profile.production_rate_per_hour * downtime_hours
        cascade_cost = self.cascade_model.calculate_cascade_cost(
            machine_profile, downtime_hours
        )
        collateral_stress = (
            COLLATERAL_STRESS_FACTOR
            * (self.planned_labor_cost + self.planned_parts_cost)
            * len(machine_profile.cascade_targets)
        )
        return repair_cost + production_loss + cascade_cost + collateral_stress

    def _attach_expected_cost(
        self,
        result: DecisionResult,
        machine_profile: MachineProfile,
        rul_hours: float,
    ) -> DecisionResult:
        """Price the failure risk each scenario leaves open."""
        if not result.is_valid:
            return result
        rtf = self.run_to_failure_cost(machine_profile)
        horizon = DECISION_HORIZON_HOURS

        if result.scenario == DecisionScenario.OBSERVE:
            p_fail = self.survival_model.failure_within(rul_hours, horizon)
            result.failure_probability = p_fail
            result.expected_cost = p_fail * rtf
        elif result.scenario == DecisionScenario.REDUCE_LOAD:
            # Slower wear extends the remaining life: RUL / wear factor.
            extended_rul = rul_hours / max(result.wear_reduction_factor, 1e-6)
            p_fail = self.survival_model.failure_within(extended_rul, horizon)
            production_loss_horizon = result.production_loss * horizon
            result.failure_probability = p_fail
            result.expected_cost = production_loss_horizon + p_fail * rtf
        elif result.scenario == DecisionScenario.PLANNED:
            # Residual risk: machine must hold until the planned slot at the
            # next shift boundary; the shift-change window is non-production
            # time, so that part of the downtime is free.
            p_fail = self.survival_model.failure_within(rul_hours, PLANNED_LEAD_HOURS)
            billable_downtime = max(
                DEFAULT_PLANNED_DOWNTIME_HOURS - SHIFT_CHANGE_OVERLAP_HOURS, 0.0
            )
            planned_cost = (
                self.planned_labor_cost
                + self.planned_parts_cost
                + machine_profile.production_rate_per_hour * billable_downtime
            )
            result.failure_probability = p_fail
            result.expected_cost = planned_cost + p_fail * rtf
        elif result.scenario == DecisionScenario.SHUTDOWN:
            # Controlled stop now — no remaining failure risk.
            result.failure_probability = 0.0
            result.expected_cost = result.cost
        return result

    def evaluate(
        self,
        scenario: DecisionScenario,
        machine_profile: MachineProfile,
        rul_hours: float = 24.0,
        load_reduction_percent: float = 20.0,
    ) -> DecisionResult:
        """
        Evaluate a decision scenario for a machine.

        Args:
            scenario: Which scenario to evaluate
            machine_profile: Machine physics profile
            rul_hours: Remaining useful life in hours
            load_reduction_percent: For REDUCE_LOAD scenario

        Returns:
            DecisionResult with costs and validity
        """
        if scenario == DecisionScenario.OBSERVE:
            result = self._evaluate_observe(machine_profile)
        elif scenario == DecisionScenario.REDUCE_LOAD:
            result = self._evaluate_reduce_load(
                machine_profile, load_reduction_percent
            )
        elif scenario == DecisionScenario.PLANNED:
            result = self._evaluate_planned(machine_profile, rul_hours)
        elif scenario == DecisionScenario.SHUTDOWN:
            result = self._evaluate_shutdown(machine_profile, rul_hours)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        return self._attach_expected_cost(result, machine_profile, rul_hours)

    def _evaluate_observe(self, machine_profile: MachineProfile) -> DecisionResult:
        """OBSERVE = 0 TL always (Murat's rule #1)."""
        return DecisionResult(
            scenario=DecisionScenario.OBSERVE,
            cost=0.0,
            is_valid=True,
            production_loss=0.0,
            cascade_cost=0.0,
            labor_cost=0.0,
            emergency_cost=0.0,
            parts_cost=0.0,
            survival_probability=1.0,
            wear_reduction_factor=1.0,
            cost_rate=0.0,
            prescriptive_action=PrescriptiveAction(
                action_type="MONITOR",
                description="Continue monitoring, no physical intervention",
            ),
        )

    def _evaluate_reduce_load(
        self,
        machine_profile: MachineProfile,
        load_reduction_percent: float,
    ) -> DecisionResult:
        """
        REDUCE_LOAD: Effective only when Weibull beta > 1.
        Rejected for Stage IV bearing damage.
        """
        # Check Stage IV bearing rejection
        if (STAGE_IV_BEARING_REJECTION and
                machine_profile.bearing_stage == "IV"):
            return DecisionResult(
                scenario=DecisionScenario.REDUCE_LOAD,
                cost=0.0,
                is_valid=False,
                rejection_reason="Stage IV bearing damage - REDUCE_LOAD ineffective",
                prescriptive_action=None,
            )

        # Check Weibull beta > 1 (wear-out dominant)
        if machine_profile.weibull_beta <= WEIBULL_EFFECTIVE_THRESHOLD:
            return DecisionResult(
                scenario=DecisionScenario.REDUCE_LOAD,
                cost=0.0,
                is_valid=False,
                rejection_reason=(
                    f"Weibull beta={machine_profile.weibull_beta} <= "
                    f"{WEIBULL_EFFECTIVE_THRESHOLD} - load reduction ineffective "
                    "for random failure mode"
                ),
                prescriptive_action=None,
            )

        # Calculate wear reduction factor
        # wear ~ load^beta, so reducing load by x% reduces wear by (1-x/100)^beta
        load_factor = 1.0 - (load_reduction_percent / 100.0)
        wear_reduction_factor = load_factor ** machine_profile.weibull_beta

        # Cost = partial production loss (not full stop)
        production_loss = (
            machine_profile.production_rate_per_hour *
            (load_reduction_percent / 100.0)
        )

        return DecisionResult(
            scenario=DecisionScenario.REDUCE_LOAD,
            cost=production_loss,
            is_valid=True,
            production_loss=production_loss,
            cascade_cost=0.0,
            labor_cost=0.0,
            emergency_cost=0.0,
            parts_cost=0.0,
            survival_probability=1.0,
            wear_reduction_factor=wear_reduction_factor,
            cost_rate=production_loss,
            prescriptive_action=PrescriptiveAction(
                action_type="SLOWDOWN",
                description=f"Reduce load by {load_reduction_percent}%",
                parameters={"load_reduction_percent": load_reduction_percent},
            ),
        )

    def _evaluate_planned(
        self,
        machine_profile: MachineProfile,
        rul_hours: float,
    ) -> DecisionResult:
        """
        PLANNED: Cost-optimal block replacement model.
        E[cost] = C_plan * (1/RUL) + C_unplanned * (1 - R(t))

        Includes production loss during planned downtime.
        """
        labor_cost = self.planned_labor_cost
        parts_cost = self.planned_parts_cost

        # Production loss during planned replacement downtime
        production_loss = (
            machine_profile.production_rate_per_hour * DEFAULT_PLANNED_DOWNTIME_HOURS
        )
        total_planned_cost = labor_cost + parts_cost + production_loss

        # Optimal replacement time (simplified: use RUL as window)
        optimal_replacement_hours = max(rul_hours * 0.8, 1.0)

        # Cost rate: total cost / time window
        # Longer RUL → lower cost rate (more time to plan)
        if rul_hours > 0:
            cost_rate = total_planned_cost / rul_hours
        else:
            cost_rate = total_planned_cost

        return DecisionResult(
            scenario=DecisionScenario.PLANNED,
            cost=total_planned_cost,
            is_valid=True,
            production_loss=production_loss,
            cascade_cost=0.0,  # Planned = no cascade risk
            labor_cost=labor_cost,
            emergency_cost=0.0,
            parts_cost=parts_cost,
            survival_probability=1.0,
            wear_reduction_factor=1.0,
            optimal_replacement_hours=optimal_replacement_hours,
            cost_rate=cost_rate,
            prescriptive_action=PrescriptiveAction(
                action_type="SCHEDULE_REPLACEMENT",
                description=f"Schedule replacement within {optimal_replacement_hours:.0f} hours",
                parameters={"optimal_hours": optimal_replacement_hours},
            ),
        )

    def _evaluate_shutdown(
        self,
        machine_profile: MachineProfile,
        rul_hours: float,
    ) -> DecisionResult:
        """
        SHUTDOWN: Safe emergency stop.

        Always offered: excluding it below a survival threshold produced a
        perverse recommendation at RUL→0 — with SHUTDOWN gone, OBSERVE (the
        full run-to-failure expectation) became the cheapest remaining
        option and the engine effectively advised watching the machine die.
        A controlled stop attempt dominates run-to-failure even when the
        survival margin is thin; the margin is annotated instead.
        """
        # Calculate survival probability
        survival_prob = self.survival_model.survival_probability(
            rul_hours, machine_profile.weibull_eta
        )
        urgent = survival_prob < SHUTDOWN_SURVIVAL_THRESHOLD

        # Calculate costs
        production_loss = (
            machine_profile.production_rate_per_hour * self.shutdown_hours
        )
        emergency_cost = production_loss * (self.emergency_multiplier - 1.0)

        # Cascade cost
        cascade_cost = self.cascade_model.calculate_cascade_cost(
            machine_profile, self.shutdown_hours
        )

        total_cost = production_loss + emergency_cost + cascade_cost

        return DecisionResult(
            scenario=DecisionScenario.SHUTDOWN,
            cost=total_cost,
            is_valid=True,
            production_loss=production_loss,
            cascade_cost=cascade_cost,
            labor_cost=0.0,
            emergency_cost=emergency_cost,
            parts_cost=0.0,
            survival_probability=survival_prob,
            wear_reduction_factor=1.0,
            cost_rate=total_cost / max(self.shutdown_hours, 1.0),
            prescriptive_action=PrescriptiveAction(
                action_type="EMERGENCY_STOP",
                description=(
                    "Execute emergency shutdown IMMEDIATELY — survival margin critical"
                    if urgent
                    else "Execute emergency shutdown"
                ),
                parameters={"shutdown_hours": self.shutdown_hours},
            ),
        )

    def recommend(
        self,
        machine_profile: MachineProfile,
        rul_hours: float = 24.0,
        load_reduction_percent: float = 20.0,
    ) -> list[RecommendedOption]:
        """
        Generate all valid recommendations sorted by risk-adjusted
        expected cost.

        Direct cost alone would always favour OBSERVE (it costs nothing
        today); the expected cost additionally prices the failure risk each
        option leaves open, so the recommendation shifts from OBSERVE to
        PLANNED/SHUTDOWN as the RUL shrinks.
        """
        scenarios = [
            DecisionScenario.OBSERVE,
            DecisionScenario.REDUCE_LOAD,
            DecisionScenario.PLANNED,
            DecisionScenario.SHUTDOWN,
        ]

        options = []
        for scenario in scenarios:
            chosen_pct = None
            survival_to_repair = None
            if scenario == DecisionScenario.REDUCE_LOAD:
                result, chosen_pct, survival_to_repair = self._best_load_reduction(
                    machine_profile, rul_hours
                )
            else:
                result = self.evaluate(
                    scenario=scenario,
                    machine_profile=machine_profile,
                    rul_hours=rul_hours,
                    load_reduction_percent=load_reduction_percent,
                )
            if result.is_valid:
                options.append(RecommendedOption(
                    scenario=scenario,
                    cost=result.cost,
                    is_recommended=False,
                    is_valid=True,
                    expected_cost=result.expected_cost,
                    failure_probability=result.failure_probability,
                    load_reduction_percent=chosen_pct,
                    survival_to_repair=survival_to_repair,
                ))

        # Sort by risk-adjusted expected cost (ascending)
        options.sort(key=lambda o: o.expected_cost)

        # Mark the lowest expected cost as recommended
        if options:
            options[0].is_recommended = True

        return options

    # Candidate load reductions, mildest first; the bridge the machine must
    # survive is the planned-repair lead plus the repair itself.
    LOAD_REDUCTION_CANDIDATES = (20.0, 40.0, 60.0)
    BRIDGE_FAILURE_TOLERANCE = 0.05

    def _best_load_reduction(
        self,
        machine_profile: MachineProfile,
        rul_hours: float,
    ) -> tuple[DecisionResult, float | None, float | None]:
        """Pick the MINIMAL load reduction that safely bridges to repair.

        The operator's question is "how little can I slow production and
        still make it to the planned slot?" — so candidates are tried
        mildest-first and the first one whose failure probability over the
        bridge window stays within tolerance wins. If even the deepest
        reduction cannot bridge, it is still returned (best effort), with
        its honest survival number.
        """
        bridge_hours = PLANNED_LEAD_HOURS + DEFAULT_PLANNED_DOWNTIME_HOURS

        best: tuple[DecisionResult, float, float] | None = None
        for pct in self.LOAD_REDUCTION_CANDIDATES:
            result = self.evaluate(
                scenario=DecisionScenario.REDUCE_LOAD,
                machine_profile=machine_profile,
                rul_hours=rul_hours,
                load_reduction_percent=pct,
            )
            if not result.is_valid:
                return result, None, None
            extended_rul = rul_hours / max(result.wear_reduction_factor, 1e-6)
            p_fail = self.survival_model.failure_within(extended_rul, bridge_hours)
            survival = 1.0 - p_fail
            best = (result, pct, survival)
            if p_fail <= self.BRIDGE_FAILURE_TOLERANCE:
                break

        result, pct, survival = best
        return result, pct, survival
