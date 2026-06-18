"""FastAPI dependencies."""

from functools import lru_cache

from api.config import get_settings
from api.services import ForecastService


@lru_cache(maxsize=1)
def get_forecast_service():
    """Return a cached forecasting service instance."""
    settings = get_settings()
    return ForecastService(
        project_root=settings.project_root,
        max_prediction_records=settings.max_prediction_records,
    )
