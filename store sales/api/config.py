"""API configuration helpers."""

from functools import lru_cache
import os
from pathlib import Path


class Settings:
    """Runtime settings for the forecasting API."""

    api_title = "Store Sales Forecasting API"
    api_version = "0.1.0"
    api_description = (
        "Batch prediction API for the Corporacion Favorita store sales "
        "forecasting model."
    )

    def __init__(self):
        self.project_root = self._read_project_root()
        self.max_prediction_records = int(
            os.getenv("STORE_SALES_MAX_PREDICTION_RECORDS", "30000")
        )

    @staticmethod
    def _read_project_root():
        configured_root = os.getenv("STORE_SALES_PROJECT_ROOT")

        if configured_root:
            return Path(configured_root).resolve()

        return None


@lru_cache(maxsize=1)
def get_settings():
    """Return cached API settings."""
    return Settings()
