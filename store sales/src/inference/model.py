"""
LightGBM model loading and validation utilities.
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.utils.validation import build_check, checks_to_dataframe


def load_lightgbm_model(model_path):
    """
    Load a LightGBM Booster from disk.

    Parameters
    ----------
    model_path : str or pathlib.Path
        LightGBM model file path.

    Returns
    -------
    lightgbm.Booster
        Loaded LightGBM Booster.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"LightGBM model file was not found: {model_path.as_posix()}"
        )

    return lgb.Booster(model_file=model_path.as_posix())


def get_model_metadata(model):
    """
    Extract basic metadata from a LightGBM Booster.

    Parameters
    ----------
    model : lightgbm.Booster
        Loaded LightGBM Booster.

    Returns
    -------
    dict
        Model metadata.
    """
    return {
        "num_features": model.num_feature(),
        "num_trees": model.num_trees(),
        "feature_names": model.feature_name(),
        "best_iteration": model.best_iteration,
        "current_iteration": model.current_iteration(),
    }


def validate_model_against_feature_contract(model, feature_names):
    """
    Validate that a LightGBM Booster is compatible with the feature contract.

    Parameters
    ----------
    model : lightgbm.Booster
        Loaded LightGBM Booster.
    feature_names : list of str
        Expected model feature names.

    Returns
    -------
    pandas.DataFrame
        Validation checks.
    """
    model_feature_names = model.feature_name()

    checks = []

    checks.append(
        build_check(
            check="model_feature_count_matches_contract",
            passed=model.num_feature() == len(feature_names),
            critical=True,
            details=(
                f"Model features: {model.num_feature()}; "
                f"contract features: {len(feature_names)}"
            ),
        )
    )

    checks.append(
        build_check(
            check="model_feature_names_match_contract",
            passed=list(model_feature_names) == list(feature_names),
            critical=True,
            details=(
                "Model feature names match the feature contract."
                if list(model_feature_names) == list(feature_names)
                else "Model feature names do not match the feature contract."
            ),
        )
    )

    checks.append(
        build_check(
            check="model_has_trees",
            passed=model.num_trees() > 0,
            critical=True,
            details=f"Number of trees: {model.num_trees()}",
        )
    )

    return checks_to_dataframe(checks)


def build_model_feature_comparison(model, feature_names):
    """
    Build a position-level comparison between model features and contract features.

    Parameters
    ----------
    model : lightgbm.Booster
        Loaded LightGBM Booster.
    feature_names : list of str
        Expected model feature names.

    Returns
    -------
    pandas.DataFrame
        Feature comparison table.
    """
    model_feature_names = model.feature_name()
    max_length = max(len(model_feature_names), len(feature_names))

    records = []

    for position in range(max_length):
        model_feature = (
            model_feature_names[position]
            if position < len(model_feature_names)
            else None
        )
        contract_feature = (
            feature_names[position]
            if position < len(feature_names)
            else None
        )

        records.append(
            {
                "feature_position": position,
                "model_feature": model_feature,
                "contract_feature": contract_feature,
                "matches": model_feature == contract_feature,
            }
        )

    return pd.DataFrame(records)
