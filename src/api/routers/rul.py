"""RUL (Remaining Useful Life) prediction endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/rul", tags=["rul"])


class RULPredictionRequest(BaseModel):
    """Request model for RUL prediction."""
    machine_id: str
    features: dict[str, float]
    phase: str
    emergency_stop_count: int = 0


class RULPredictionResponse(BaseModel):
    """Response model for RUL prediction."""
    rul_hours: float
    rul_low_ci: float
    rul_high_ci: float
    confidence: float
    failure_prob_24h: float
    survive_shift_pct: float
    method: str
    fallback: bool
    model_trained: bool


@router.post("/predict", response_model=RULPredictionResponse | None)
async def predict_rul(request: RULPredictionRequest):
    """Predict remaining useful life for a machine.

    Args:
        request: Machine ID, features, phase, and emergency stop count.

    Returns:
        RUL prediction with confidence intervals, or None if phase is HEALTHY.
    """
    try:
        from src.ml.rul_predictor import RULPredictor

        predictor = RULPredictor(machine_id=request.machine_id)
        result = predictor.predict(
            features=request.features,
            phase=request.phase,
            emergency_stop_count=request.emergency_stop_count
        )

        if result is None:
            return None

        return RULPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RUL prediction failed: {str(e)}") from e
