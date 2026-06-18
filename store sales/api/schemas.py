"""Pydantic schemas for the forecasting API."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str
    service: str
    version: str
    model_loaded: bool
    artifact_checks: list[dict[str, Any]]


class FeatureContractResponse(BaseModel):
    """Model feature contract response."""

    n_features: int
    feature_names: list[str]
    categorical_features: list[str]
    prediction_grain: list[str]


class ModelMetadataResponse(BaseModel):
    """Model metadata response."""

    recommended_model_version: str
    serving_model_version: str
    uses_full_serving_model: bool
    model_path: str
    config_path: str
    n_features: int
    model_metadata: dict[str, Any]


class PredictionRequest(BaseModel):
    """Batch prediction request using model-ready feature records."""

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Model-ready feature records. Each record must include the model "
            "feature contract plus the prediction grain columns."
        ),
    )
    include_diagnostics: bool = Field(
        default=False,
        description="Include validation diagnostics in the response.",
    )


class PredictionResponse(BaseModel):
    """Batch prediction response."""

    rows: int
    predictions: list[dict[str, Any]]
    summary: dict[str, Any]
    diagnostics: dict[str, Any] | None = None
