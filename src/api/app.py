"""FastAPI application for PDM Intelligence v3.0.0"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import live
from src.api.routers import (
    alarms,
    anomaly,
    audit,
    cost_optimizer,
    decisions,
    fleet,
    health,
    knowledge_graph,
    machines,
    rul,
    savings,
    shift_reports,
    work_orders,
)
from src.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop the live-data broadcast tasks with the app."""
    tasks = [
        asyncio.create_task(live.snapshot_loop()),
        asyncio.create_task(live.anomaly_forward_loop()),
    ]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(
    title="PDM Intelligence API",
    description="Industrial AI Predictive Maintenance System API",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS (origins restricted via env in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inference endpoints (stateless model wrappers)
app.include_router(health.router)
app.include_router(anomaly.router)
app.include_router(rul.router)
app.include_router(knowledge_graph.router)
app.include_router(cost_optimizer.router)

# Dashboard endpoints (database-backed)
API_V1_PREFIX = "/api/v1"
app.include_router(machines.router, prefix=API_V1_PREFIX)
app.include_router(fleet.router, prefix=API_V1_PREFIX)
app.include_router(alarms.router, prefix=API_V1_PREFIX)
app.include_router(decisions.router, prefix=API_V1_PREFIX)
app.include_router(audit.router, prefix=API_V1_PREFIX)
app.include_router(work_orders.router, prefix=API_V1_PREFIX)
app.include_router(savings.router, prefix=API_V1_PREFIX)
app.include_router(shift_reports.router, prefix=API_V1_PREFIX)

# Live data over WebSocket
app.add_api_websocket_route("/ws/live", live.websocket_endpoint)


@app.get("/")
async def root():
    """Root endpoint with API information.

    Returns:
        dict: API name, version, and documentation links.
    """
    return {
        "name": "PDM Intelligence API",
        "version": "3.0.0",
        "description": "Industrial AI Predictive Maintenance System",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "api": API_V1_PREFIX,
        "ws": "/ws/live",
    }
