"""Health check routes."""

from fastapi import APIRouter, Depends

from api.config import get_settings
from api.dependencies import get_forecast_service
from api.schemas import HealthResponse, ReadinessResponse
from api.services import ForecastService


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Return a lightweight liveness check."""
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.api_title,
        "version": settings.api_version,
    }


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness_check(service: ForecastService = Depends(get_forecast_service)):
    """Return model and artifact readiness."""
    settings = get_settings()
    readiness = service.readiness()

    return {
        "status": "ready" if readiness["ready"] else "not_ready",
        "service": settings.api_title,
        "version": settings.api_version,
        "model_loaded": readiness["model_loaded"],
        "artifact_checks": readiness["artifact_checks"],
    }
