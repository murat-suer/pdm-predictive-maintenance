"""Knowledge Graph Root Cause Analysis endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


class RootCauseResponse(BaseModel):
    """Response model for root cause analysis."""
    sensor: str
    weight: float
    rank: int


@router.get("/root-causes/{failure_mode}", response_model=list[RootCauseResponse])
async def get_root_causes(failure_mode: str, top_k: int = 5):
    """Get ranked root causes (sensors) for a failure mode.

    Args:
        failure_mode: The failure mode to analyze.
        top_k: Maximum number of root causes to return.

    Returns:
        List of sensors ranked by causal weight.
    """
    try:
        from src.ml.causal_graph import CausalKnowledgeGraph

        kg = CausalKnowledgeGraph()
        causes = kg.find_root_causes(failure_mode=failure_mode, top_k=top_k)

        # find_root_causes returns list[dict] with sensor, weight, rank fields
        result = []
        for cause in causes:
            result.append(RootCauseResponse(
                sensor=cause["sensor"],
                weight=cause["weight"],
                rank=cause["rank"]
            ))
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Root cause analysis failed: {str(e)}") from e
