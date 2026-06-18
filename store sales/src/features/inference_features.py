"""
Inference feature loading utilities.

This module loads model-ready feature tables for batch inference.
"""

from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, dataframe_memory_mb


def load_model_ready_inference_features(features_path, parse_dates=True):
    """
    Load model-ready batch inference features.

    Parameters
    ----------
    features_path : str or pathlib.Path
        Path to the model-ready inference feature parquet file.
    parse_dates : bool
        Whether to parse the date column as datetime.

    Returns
    -------
    pandas.DataFrame
        Model-ready inference feature dataframe.
    """
    features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Inference features file was not found: {features_path.as_posix()}"
        )

    inference_features = read_parquet(features_path)

    if parse_dates and "date" in inference_features.columns:
        inference_features = inference_features.copy()
        inference_features["date"] = pd.to_datetime(inference_features["date"])

    return inference_features


def build_feature_load_summary(dataframe, source_path):
    """
    Build a summary of the loaded inference features.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Loaded inference feature dataframe.
    source_path : str or pathlib.Path
        Source file path.

    Returns
    -------
    dict
        Feature loading summary.
    """
    source_path = Path(source_path)

    if "date" in dataframe.columns:
        min_date = dataframe["date"].min()
        max_date = dataframe["date"].max()
        n_dates = dataframe["date"].nunique()
    else:
        min_date = None
        max_date = None
        n_dates = None

    return {
        "source_path": source_path.as_posix(),
        "rows": len(dataframe),
        "columns": dataframe.shape[1],
        "memory_mb": dataframe_memory_mb(dataframe),
        "min_date": min_date,
        "max_date": max_date,
        "n_dates": n_dates,
    }


def get_prediction_grain_columns(default_grain=None):
    """
    Return the default prediction grain columns.

    Parameters
    ----------
    default_grain : list of str, optional
        Custom grain definition.

    Returns
    -------
    list of str
        Prediction grain columns.
    """
    if default_grain is None:
        return ["date", "store_nbr", "family"]

    return list(default_grain)


def build_prediction_grain_summary(dataframe, grain_columns):
    """
    Build a summary of the inference prediction grain.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Inference dataframe.
    grain_columns : list of str
        Columns defining prediction grain.

    Returns
    -------
    dict
        Prediction grain summary.
    """
    missing_grain_columns = sorted(set(grain_columns) - set(dataframe.columns))

    if missing_grain_columns:
        return {
            "grain_columns": grain_columns,
            "grain_available": False,
            "missing_grain_columns": missing_grain_columns,
            "rows": len(dataframe),
            "unique_grain_rows": None,
            "duplicated_grain_rows": None,
        }

    unique_grain_rows = dataframe[grain_columns].drop_duplicates().shape[0]
    duplicated_grain_rows = len(dataframe) - unique_grain_rows

    return {
        "grain_columns": grain_columns,
        "grain_available": True,
        "missing_grain_columns": [],
        "rows": len(dataframe),
        "unique_grain_rows": unique_grain_rows,
        "duplicated_grain_rows": duplicated_grain_rows,
    }
