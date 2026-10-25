"""
Root API router — aggregates all sub-routers.
"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.api.swap import router as swap_router

api_router = APIRouter()

# Observability (no prefix — /health and /ready live at root)
api_router.include_router(health_router)

# Jobs API — versioned under /api/v1
api_router.include_router(jobs_router, prefix="/api/v1", tags=["jobs"])

# Swap endpoint — versioned under /api/v1
api_router.include_router(swap_router, prefix="/api/v1", tags=["jobs"])
