"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Service health check endpoint.

    Returns:
        dict: Status and version information.
    """
    return {
        "status": "healthy",
        "version": "3.0.0"
    }
