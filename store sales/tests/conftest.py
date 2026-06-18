import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_forecast_service
from api.main import create_app
from api.services import ForecastService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def forecast_service(project_root):
    os.environ["STORE_SALES_PROJECT_ROOT"] = project_root.as_posix()
    return ForecastService(project_root=project_root)


@pytest.fixture(scope="session")
def sample_feature_dataframe(project_root):
    feature_path = project_root / "data" / "features" / "baseline_test_features.parquet"
    return pd.read_parquet(feature_path).head(3)


@pytest.fixture(scope="session")
def sample_feature_records(sample_feature_dataframe):
    payload = sample_feature_dataframe.to_json(orient="records", date_format="iso")
    return json.loads(payload)


@pytest.fixture()
def client(forecast_service):
    app = create_app()
    app.dependency_overrides[get_forecast_service] = lambda: forecast_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()

