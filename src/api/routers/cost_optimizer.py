"""Cost optimization endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/cost-optimize", tags=["cost-optimizer"])


class CostOptimizeRequest(BaseModel):
    """Request model for cost optimization."""
    eta: float = Field(..., description="Weibull scale parameter (characteristic life)")
    beta: float = Field(..., description="Weibull shape parameter")
    preventive_cost: float = Field(default=1000.0, description="Cost of preventive maintenance")
    corrective_cost: float = Field(default=5000.0, description="Cost of corrective maintenance")
    downtime_cost_per_hour: float = Field(default=500.0, description="Downtime cost per hour")
    production_loss_per_hour: float = Field(default=2000.0, description="Production loss per hour")


class CostOptimizeResponse(BaseModel):
    """Response model for cost optimization."""
    optimal_tp: float
    min_cost_rate: float
    strategy_comparison: dict[str, float]


@router.post("", response_model=CostOptimizeResponse)
async def optimize_cost(request: CostOptimizeRequest):
    """Find optimal preventive maintenance timing.

    Args:
        request: Weibull parameters and cost factors.

    Returns:
        Optimal replacement time and cost analysis.
    """
    try:
        from src.ml.maintenance_optimizer import MaintenanceCostOptimizer

        optimizer = MaintenanceCostOptimizer(
            preventive_cost=request.preventive_cost,
            corrective_cost=request.corrective_cost,
            downtime_cost_per_hour=request.downtime_cost_per_hour,
            production_loss_per_hour=request.production_loss_per_hour
        )

        optimal_tp, min_cost_rate = optimizer.find_optimal_replacement_time(
            eta=request.eta,
            beta=request.beta
        )

        strategy_comparison = optimizer.compare_strategies(
            eta=request.eta,
            beta=request.beta,
            tp_optimal=optimal_tp
        )

        return CostOptimizeResponse(
            optimal_tp=optimal_tp,
            min_cost_rate=min_cost_rate,
            strategy_comparison=strategy_comparison
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost optimization failed: {str(e)}") from e
