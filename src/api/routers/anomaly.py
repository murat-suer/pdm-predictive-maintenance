"""Anomaly detection endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


class AnomalyPredictionRequest(BaseModel):
    """Request model for anomaly prediction."""
    machine_id: str
    features: dict[str, float]


class AnomalyPredictionResponse(BaseModel):
    """Response model for anomaly prediction."""
    is_anomaly: bool
    anomaly_score: float
    top_contributing_sensor: str | None = None
    shap_values: dict[str, float] = {}


@router.post("/predict", response_model=AnomalyPredictionResponse)
async def predict_anomaly(request: AnomalyPredictionRequest):
    """Predict anomaly for given features.

    Args:
        request: Machine ID and feature dictionary.

    Returns:
        Anomaly prediction with score and contributing sensors.
    """
    try:
        from src.ml.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(machine_id=request.machine_id)
        result = detector.predict(request.features)

        return AnomalyPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly prediction failed: {str(e)}") from e
