"""Forecasting routes."""

from fastapi import APIRouter, Depends

from api.dependencies import get_forecast_service
from api.schemas import (
    FeatureContractResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.services import ForecastService


router = APIRouter()


@router.get("/features", response_model=FeatureContractResponse)
def get_feature_contract(service: ForecastService = Depends(get_forecast_service)):
    """Return the model feature contract expected by the prediction endpoint."""
    return service.feature_contract_response()


@router.get("/model", response_model=ModelMetadataResponse)
def get_model_metadata(service: ForecastService = Depends(get_forecast_service)):
    """Return selected serving model metadata."""
    return service.model_metadata_response()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    service: ForecastService = Depends(get_forecast_service),
):
    """Generate sales predictions from model-ready feature records."""
    return service.predict_records(
        records=request.records,
        include_diagnostics=request.include_diagnostics,
    )
