"""
Honest cost-savings calculator for PDM v3.

KRİTİK KURAL: OBSERVATION = 0 TL (ASLA pozitif olmamalı)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Decision types for maintenance actions."""
    OBSERVATION = "OBSERVATION"
    TECHNICAL_DISPATCH = "TECHNICAL_DISPATCH"
    SLOWDOWN_ORDER = "SLOWDOWN_ORDER"
    STOP_PREP_ORDER = "STOP_PREP_ORDER"
    CONTROLLED_STOP = "CONTROLLED_STOP"


@dataclass
class CostConfig:
    """Configuration for cost calculations."""
    production_rate_per_hour: float = 5000.0
    emergency_response_cost: float = 10000.0
    cascade_multiplier: float = 1.5
    premium_factor: float = 1.2
    wear_reduction_factor: float = 0.3
    machines: dict[str, dict] = field(default_factory=dict)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a decision."""
    total: float = 0.0
    production_loss: float = 0.0
    emergency_cost: float = 0.0
    cascade_risk: float = 0.0
    labor_cost: float = 0.0
    savings: float = 0.0


class CostCalculator:
    """
    Calculates costs for different maintenance decisions.

    KRİTİK: OBSERVATION = 0 TL (ASLA pozitif olmamalı)
    """

    def __init__(self, config_path: str | None = None):
        self.config = self._load_config(config_path)
        self._consecutive_observe_count = 0

    def _load_config(self, config_path: str | None) -> CostConfig:
        """Load config from YAML file or use defaults."""
        if config_path is None:
            return CostConfig()

        path = Path(config_path)
        if not path.exists():
            return CostConfig()

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            return CostConfig(
                production_rate_per_hour=float(data.get("production_rate_per_hour", 5000.0)),
                emergency_response_cost=float(data.get("emergency_response_cost", 10000.0)),
                cascade_multiplier=float(data.get("cascade_multiplier", 1.5)),
                premium_factor=float(data.get("premium_factor", 1.2)),
                wear_reduction_factor=float(data.get("wear_reduction_factor", 0.3)),
                machines=data.get("machines", {}),
            )
        except Exception:
            return CostConfig()

    def _get_production_rate(self, machine_id: str) -> float:
        """Get production rate for a specific machine."""
        if machine_id and machine_id in self.config.machines:
            machine_config = self.config.machines[machine_id]
            if isinstance(machine_config, dict):
                return float(machine_config.get("production_rate_per_hour",
                                                 self.config.production_rate_per_hour))
        return self.config.production_rate_per_hour

    def calculate(
        self,
        decision_type,
        machine_id: str,
        rul_hours: float | None = None,
        observation_minutes: float | None = None,
    ) -> CostBreakdown:
        """
        Calculate cost breakdown for a decision.

        KRİTİK: OBSERVATION = 0 TL (ASLA pozitif olmamalı)
        """
        # Handle None or unknown decision types
        if decision_type is None:
            return CostBreakdown(total=0.0)

        # Convert string to enum if needed
        if isinstance(decision_type, str):
            try:
                decision_type = DecisionType(decision_type)
            except ValueError:
                return CostBreakdown(total=0.0)

        # KRİTİK: OBSERVATION = 0 TL
        if decision_type == DecisionType.OBSERVATION:
            self._consecutive_observe_count += 1
            if self._consecutive_observe_count >= 3:
                logger.warning(
                    "Anti-pattern detected: %d+ consecutive OBSERVATION recommendations",
                    self._consecutive_observe_count,
                )
            return CostBreakdown(
                total=0.0,
                production_loss=0.0,
                emergency_cost=0.0,
                cascade_risk=0.0,
                labor_cost=0.0,
                savings=0.0,
            )

        rate = self._get_production_rate(machine_id)
        # Non-OBSERVATION decision resets the consecutive counter
        self._consecutive_observe_count = 0

        if decision_type == DecisionType.TECHNICAL_DISPATCH:
            # Emergency response + cascade risk
            emergency = self.config.emergency_response_cost
            cascade = rate * self.config.cascade_multiplier
            total = emergency + cascade
            return CostBreakdown(
                total=total,
                production_loss=0.0,
                emergency_cost=emergency,
                cascade_risk=cascade,
                labor_cost=0.0,
            )

        elif decision_type == DecisionType.SLOWDOWN_ORDER:
            # Production loss with wear reduction
            production_loss = rate * self.config.wear_reduction_factor
            return CostBreakdown(
                total=production_loss,
                production_loss=production_loss,
                emergency_cost=0.0,
                cascade_risk=0.0,
                labor_cost=0.0,
            )

        elif decision_type == DecisionType.STOP_PREP_ORDER:
            # Full production loss
            production_loss = rate
            return CostBreakdown(
                total=production_loss,
                production_loss=production_loss,
                emergency_cost=0.0,
                cascade_risk=0.0,
                labor_cost=0.0,
            )

        elif decision_type == DecisionType.CONTROLLED_STOP:
            # Cascade * premium
            cascade = rate * self.config.cascade_multiplier
            premium = cascade * self.config.premium_factor
            total = premium
            return CostBreakdown(
                total=total,
                production_loss=0.0,
                emergency_cost=0.0,
                cascade_risk=cascade,
                labor_cost=0.0,
            )

        # Unknown type
        return CostBreakdown(total=0.0)

    def reset_observe_counter(self) -> None:
        """Reset the consecutive OBSERVATION counter."""
        self._consecutive_observe_count = 0

    @property
    def consecutive_observe_count(self) -> int:
        """Current consecutive OBSERVATION count."""
        return self._consecutive_observe_count
