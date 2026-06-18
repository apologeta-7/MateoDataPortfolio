"""Forecasting service used by the FastAPI routes."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import status
import numpy as np
import pandas as pd

from api.exceptions import ApiError, UNPROCESSABLE_STATUS_CODE
from src.config.inference import (
    PREDICTION_COLUMN,
    PREDICTION_GRAIN,
    get_inference_artifact_paths,
)
from src.config.paths import get_project_paths, relative_to_project
from src.features.categorical import (
    apply_categorical_schema,
    load_categorical_reference_categories,
    validate_categorical_alignment,
)
from src.inference.artifacts import (
    build_artifact_inventory,
    extract_recommended_model,
    load_feature_contract,
    load_serving_metadata,
    select_serving_model,
    validate_required_artifacts,
)
from src.inference.model import (
    get_model_metadata,
    load_lightgbm_model,
    validate_model_against_feature_contract,
)
from src.inference.postprocess import (
    build_prediction_output,
    build_prediction_summary,
)
from src.inference.predict import generate_predictions
from src.inference.validation import (
    make_model_input_matrix,
    validate_feature_contract,
    validate_prediction_output,
)
from src.utils.validation import validate_no_critical_failures


class ForecastService:
    """Load model artifacts and generate store sales predictions."""

    def __init__(self, project_root=None, max_prediction_records=30000):
        self.max_prediction_records = int(max_prediction_records)
        self.paths = get_project_paths(project_root=project_root)
        self.project_root = Path(self.paths["project_root"]).resolve()
        self.artifact_paths = get_inference_artifact_paths(self.paths)

        self.artifact_inventory = build_artifact_inventory(self.artifact_paths)
        self.artifact_validation = validate_required_artifacts(
            self.artifact_inventory,
            required_for="pipeline",
        )
        validate_no_critical_failures(self.artifact_validation)

        self.recommendation = extract_recommended_model(
            self.artifact_paths["final_model_recommendation"]
        )
        self.serving_selection = select_serving_model(
            artifact_paths=self.artifact_paths,
            recommended_model_version=self.recommendation[
                "recommended_model_version"
            ],
            prefer_full_model=True,
        )
        self.serving_metadata = load_serving_metadata(self.serving_selection)

        self.feature_contract = load_feature_contract(
            self.artifact_paths["lightgbm_feature_list"]
        )
        self.feature_names = self.feature_contract["feature_names"]

        self.model = load_lightgbm_model(self.serving_selection["model_path"])
        self.model_metadata = get_model_metadata(self.model)

        self.model_validation = validate_model_against_feature_contract(
            model=self.model,
            feature_names=self.feature_names,
        )
        validate_no_critical_failures(self.model_validation)

        self.categorical_features = self._read_categorical_features()
        self.categorical_schema = load_categorical_reference_categories(
            reference_paths=[
                self.artifact_paths["baseline_train_features"],
                self.artifact_paths["baseline_valid_features"],
                self.artifact_paths["baseline_test_features"],
            ],
            categorical_features=self.categorical_features,
        )

    def readiness(self):
        """Return model and artifact readiness details."""
        artifact_checks = self._dataframe_records(
            self.artifact_inventory[
                [
                    "artifact_key",
                    "exists",
                    "required",
                    "required_for",
                    "path",
                    "file_size_mb",
                ]
            ]
        )

        for record in artifact_checks:
            record["path"] = self._relative_path(record["path"])

        ready = (
            bool(self.model is not None)
            and bool(self.artifact_validation["passed"].all())
            and bool(self.model_validation["passed"].all())
        )

        return {
            "ready": ready,
            "model_loaded": self.model is not None,
            "artifact_checks": artifact_checks,
        }

    def feature_contract_response(self):
        """Return the model feature contract in API response format."""
        return {
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "prediction_grain": list(PREDICTION_GRAIN),
        }

    def model_metadata_response(self):
        """Return selected serving model metadata in API response format."""
        return {
            "recommended_model_version": self.serving_metadata[
                "recommended_model_version"
            ],
            "serving_model_version": self.serving_metadata[
                "serving_model_version"
            ],
            "uses_full_serving_model": self.serving_metadata[
                "uses_full_serving_model"
            ],
            "model_path": self._relative_path(self.serving_metadata["model_path"]),
            "config_path": self._relative_path(self.serving_metadata["config_path"]),
            "n_features": len(self.feature_names),
            "model_metadata": self._json_safe(self.model_metadata),
        }

    def predict_records(self, records, include_diagnostics=False):
        """Generate predictions from model-ready feature records."""
        if len(records) > self.max_prediction_records:
            raise ApiError(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                code="TOO_MANY_RECORDS",
                message=(
                    f"Request contains {len(records)} records; "
                    f"maximum allowed is {self.max_prediction_records}."
                ),
            )

        dataframe = pd.DataFrame.from_records(records)

        if dataframe.empty:
            raise ApiError(
                status_code=UNPROCESSABLE_STATUS_CODE,
                code="EMPTY_REQUEST",
                message="Prediction request must include at least one record.",
            )

        dataframe = self._prepare_input_dataframe(dataframe)
        categorical_alignment_report = pd.DataFrame()
        categorical_validation = pd.DataFrame()

        if self.categorical_features:
            dataframe, categorical_alignment_report = apply_categorical_schema(
                dataframe=dataframe,
                categorical_schema=self.categorical_schema,
            )
            categorical_validation = validate_categorical_alignment(
                categorical_alignment_report
            )
            validate_no_critical_failures(categorical_validation)

        feature_validation = validate_feature_contract(
            dataframe=dataframe,
            feature_names=self.feature_names,
        )
        validate_no_critical_failures(feature_validation)

        model_input = make_model_input_matrix(
            dataframe=dataframe,
            feature_names=self.feature_names,
        )

        prediction_result = generate_predictions(
            model=self.model,
            model_input=model_input,
            prediction_column=PREDICTION_COLUMN,
        )

        prediction_output = build_prediction_output(
            source_dataframe=dataframe,
            prediction_result=prediction_result,
            grain_columns=PREDICTION_GRAIN,
            prediction_column=PREDICTION_COLUMN,
            metadata={
                "serving_model_version": self.serving_metadata[
                    "serving_model_version"
                ],
            },
        )

        output_validation = validate_prediction_output(
            dataframe=prediction_output,
            prediction_column=PREDICTION_COLUMN,
            grain_columns=PREDICTION_GRAIN,
        )
        validate_no_critical_failures(output_validation)

        response = {
            "rows": len(prediction_output),
            "predictions": self._dataframe_records(prediction_output),
            "summary": self._json_safe(
                build_prediction_summary(
                    prediction_output,
                    prediction_column=PREDICTION_COLUMN,
                )
            ),
            "diagnostics": None,
        }

        if include_diagnostics:
            response["diagnostics"] = {
                "feature_validation": self._dataframe_records(feature_validation),
                "output_validation": self._dataframe_records(output_validation),
                "categorical_alignment": self._dataframe_records(
                    categorical_alignment_report
                ),
                "categorical_validation": self._dataframe_records(
                    categorical_validation
                ),
            }

        return response

    def _prepare_input_dataframe(self, dataframe):
        dataframe = dataframe.copy()

        missing_grain_columns = sorted(set(PREDICTION_GRAIN) - set(dataframe.columns))
        if missing_grain_columns:
            raise ApiError(
                status_code=UNPROCESSABLE_STATUS_CODE,
                code="MISSING_GRAIN_COLUMNS",
                message="Prediction records are missing required grain columns.",
                details={"missing_columns": missing_grain_columns},
            )

        missing_features = sorted(set(self.feature_names) - set(dataframe.columns))
        if missing_features:
            raise ApiError(
                status_code=UNPROCESSABLE_STATUS_CODE,
                code="MISSING_MODEL_FEATURES",
                message="Prediction records are missing required model features.",
                details={"missing_features": missing_features},
            )

        if "date" in dataframe.columns:
            try:
                dataframe["date"] = pd.to_datetime(dataframe["date"])
            except Exception as exc:
                raise ApiError(
                    status_code=UNPROCESSABLE_STATUS_CODE,
                    code="INVALID_DATE",
                    message="Column 'date' must contain parseable dates.",
                ) from exc

        duplicated_grain_rows = int(dataframe.duplicated(PREDICTION_GRAIN).sum())
        if duplicated_grain_rows > 0:
            raise ApiError(
                status_code=UNPROCESSABLE_STATUS_CODE,
                code="DUPLICATED_GRAIN",
                message="Prediction records contain duplicated prediction grain rows.",
                details={"duplicated_rows": duplicated_grain_rows},
            )

        return dataframe

    def _read_categorical_features(self):
        config = self.serving_metadata.get("config", {})
        features_config = config.get("features", {})
        categorical_features = features_config.get("categorical_features", [])

        return [
            feature
            for feature in categorical_features
            if feature in self.feature_names
        ]

    def _relative_path(self, path):
        return relative_to_project(path, project_root=self.project_root)

    def _dataframe_records(self, dataframe):
        if dataframe is None or dataframe.empty:
            return []

        records = dataframe.replace({np.nan: None}).to_dict(orient="records")
        return self._json_safe(records)

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._json_safe(item) for item in value]

        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, np.generic):
            return value.item()

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        return value
