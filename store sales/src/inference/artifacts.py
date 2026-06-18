"""
Inference artifact utilities.

This module manages artifact inventory, model selection metadata and feature contracts.
"""

from pathlib import Path

import pandas as pd

from src.utils.io import read_json, read_csv_if_exists, file_size_mb
from src.utils.validation import build_check, checks_to_dataframe


PIPELINE_REQUIRED_ARTIFACTS = {
    "baseline_test_features",
    "lightgbm_model",
    "lightgbm_config",
    "lightgbm_feature_list",
}

COMPARISON_REQUIRED_ARTIFACTS = {
    "notebook_08_predictions",
}

FEATURE_CONTRACT_KEYS = [
    "feature_names",
    "features",
    "model_features",
    "selected_features",
    "feature_list",
]


def build_artifact_inventory(artifact_paths):
    """
    Build an artifact inventory table.

    Parameters
    ----------
    artifact_paths : dict
        Dictionary of artifact names and paths.

    Returns
    -------
    pandas.DataFrame
        Artifact inventory with existence and file size metadata.
    """
    records = []

    for artifact_key, path in artifact_paths.items():
        path = Path(path)

        if artifact_key in PIPELINE_REQUIRED_ARTIFACTS:
            required_for = "pipeline"
            required = True
        elif artifact_key in COMPARISON_REQUIRED_ARTIFACTS:
            required_for = "comparison"
            required = False
        else:
            required_for = "optional"
            required = False

        records.append(
            {
                "artifact_key": artifact_key,
                "path": path.as_posix(),
                "exists": path.exists(),
                "required": required,
                "required_for": required_for,
                "file_size_mb": file_size_mb(path),
            }
        )

    return pd.DataFrame(records)


def validate_required_artifacts(artifact_inventory, required_for="pipeline"):
    """
    Validate that required artifacts are available.

    Parameters
    ----------
    artifact_inventory : pandas.DataFrame
        Artifact inventory table.
    required_for : str
        Required group to validate.

    Returns
    -------
    pandas.DataFrame
        Validation checks.
    """
    required_artifacts = artifact_inventory[
        artifact_inventory["required_for"] == required_for
    ].copy()

    missing_artifacts = required_artifacts[
        required_artifacts["exists"] == False
    ]["artifact_key"].tolist()

    checks = [
        build_check(
            check=f"{required_for}_required_artifacts_available",
            passed=len(missing_artifacts) == 0,
            critical=True,
            details=(
                "All required artifacts are available."
                if len(missing_artifacts) == 0
                else f"Missing artifacts: {missing_artifacts}"
            ),
        )
    ]

    return checks_to_dataframe(checks)


def extract_recommended_model(
    recommendation_path,
    default_model_version="lightgbm_v1",
):
    """
    Extract the recommended model version from a recommendation CSV.

    If the recommendation file is missing or does not contain an expected column,
    the default model version is returned.

    Parameters
    ----------
    recommendation_path : str or pathlib.Path
        Final model recommendation CSV path.
    default_model_version : str
        Fallback model version.

    Returns
    -------
    dict
        Recommended model metadata.
    """
    recommendation_path = Path(recommendation_path)

    if not recommendation_path.exists():
        return {
            "recommended_model_version": default_model_version,
            "source": "default",
            "recommendation_available": False,
            "path": recommendation_path.as_posix(),
            "notes": "Recommendation file was not found. Default model version was used.",
        }

    recommendation = read_csv_if_exists(recommendation_path)

    if recommendation is None or recommendation.empty:
        return {
            "recommended_model_version": default_model_version,
            "source": "default",
            "recommendation_available": False,
            "path": recommendation_path.as_posix(),
            "notes": "Recommendation file was empty. Default model version was used.",
        }

    if "is_recommended" in recommendation.columns:
        recommended_rows = recommendation[
            recommendation["is_recommended"].astype(str).str.lower().isin(["true", "1", "yes"])
        ]

        if not recommended_rows.empty:
            recommendation = recommended_rows

    candidate_columns = [
        "recommended_model_version",
        "final_model_version",
        "model_version",
        "version",
        "model_name",
        "recommended_model",
    ]

    for column in candidate_columns:
        if column in recommendation.columns:
            value = recommendation[column].dropna().astype(str).iloc[0]

            return {
                "recommended_model_version": value,
                "source": "final_model_recommendation",
                "recommendation_available": True,
                "path": recommendation_path.as_posix(),
                "notes": f"Recommendation loaded from column: {column}",
            }

    return {
        "recommended_model_version": default_model_version,
        "source": "default",
        "recommendation_available": True,
        "path": recommendation_path.as_posix(),
        "notes": "No expected model version column was found. Default model version was used.",
    }


def select_serving_model(
    artifact_paths,
    recommended_model_version="lightgbm_v1",
    prefer_full_model=True,
):
    """
    Select the model artifact to serve.

    The full serving model is preferred when available. Otherwise, the original
    LightGBM v1 model is used.

    Parameters
    ----------
    artifact_paths : dict
        Inference artifact path registry.
    recommended_model_version : str
        Recommended base model version.
    prefer_full_model : bool
        Whether to prefer the full serving model when available.

    Returns
    -------
    dict
        Serving model selection metadata.

    Raises
    ------
    FileNotFoundError
        If no usable model artifact is found.
    """
    base_model_path = Path(artifact_paths["lightgbm_model"])
    base_config_path = Path(artifact_paths["lightgbm_config"])

    full_model_path = Path(artifact_paths["lightgbm_full_model"])
    full_config_path = Path(artifact_paths["lightgbm_full_config"])

    full_model_available = full_model_path.exists()
    full_config_available = full_config_path.exists()

    if prefer_full_model and full_model_available:
        selected_model_path = full_model_path
        selected_config_path = full_config_path if full_config_available else base_config_path
        serving_model_version = "lightgbm_v1_full"
        uses_full_serving_model = True
    elif base_model_path.exists():
        selected_model_path = base_model_path
        selected_config_path = base_config_path
        serving_model_version = "lightgbm_v1"
        uses_full_serving_model = False
    else:
        raise FileNotFoundError(
            "No usable LightGBM model artifact was found. "
            f"Checked: {base_model_path.as_posix()} and {full_model_path.as_posix()}"
        )

    return {
        "recommended_model_version": recommended_model_version,
        "serving_model_version": serving_model_version,
        "model_path": selected_model_path,
        "config_path": selected_config_path,
        "uses_full_serving_model": uses_full_serving_model,
        "base_model_path": base_model_path,
        "full_model_path": full_model_path,
        "full_model_available": full_model_available,
        "full_config_available": full_config_available,
    }


def load_feature_contract(feature_contract_path):
    """
    Load the model feature contract.

    Parameters
    ----------
    feature_contract_path : str or pathlib.Path
        JSON file containing the feature list.

    Returns
    -------
    dict
        Normalized feature contract.
    """
    feature_contract_path = Path(feature_contract_path)
    raw_contract = read_json(feature_contract_path)

    if isinstance(raw_contract, list):
        feature_names = raw_contract
    elif isinstance(raw_contract, dict):
        feature_names = None

        for key in FEATURE_CONTRACT_KEYS:
            if key in raw_contract:
                feature_names = raw_contract[key]
                break

        if feature_names is None:
            raise ValueError(
                "Feature contract JSON does not contain a recognized feature list key. "
                f"Expected one of: {FEATURE_CONTRACT_KEYS}"
            )
    else:
        raise TypeError(
            "Feature contract must be either a list or a dictionary."
        )

    if not isinstance(feature_names, list):
        raise TypeError("Feature names must be stored as a list.")

    if len(feature_names) == 0:
        raise ValueError("Feature contract is empty.")

    duplicated_features = sorted(
        pd.Series(feature_names)[pd.Series(feature_names).duplicated()].unique().tolist()
    )

    if duplicated_features:
        raise ValueError(
            f"Feature contract contains duplicated features: {duplicated_features}"
        )

    return {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "source_path": feature_contract_path,
        "raw_contract": raw_contract,
    }


def load_serving_metadata(serving_selection):
    """
    Load serving model metadata from its config file when available.

    Parameters
    ----------
    serving_selection : dict
        Serving model selection metadata.

    Returns
    -------
    dict
        Serving metadata.
    """
    config_path = Path(serving_selection["config_path"])

    if config_path.exists():
        config = read_json(config_path)
        config_available = True
    else:
        config = {}
        config_available = False

    return {
        "serving_model_version": serving_selection["serving_model_version"],
        "recommended_model_version": serving_selection["recommended_model_version"],
        "uses_full_serving_model": serving_selection["uses_full_serving_model"],
        "model_path": serving_selection["model_path"].as_posix(),
        "config_path": config_path.as_posix(),
        "config_available": config_available,
        "config": config,
    }
