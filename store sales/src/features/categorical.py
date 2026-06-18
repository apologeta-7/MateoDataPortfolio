"""
Categorical feature alignment utilities.

This module helps align categorical feature dtypes before LightGBM inference.
"""

from pathlib import Path

import pandas as pd


def detect_categorical_features(dataframe, feature_names):
    """
    Detect categorical model features from a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.
    feature_names : list of str
        Model feature names.

    Returns
    -------
    list of str
        Detected categorical feature names.
    """
    categorical_features = []

    for feature_name in feature_names:
        if feature_name not in dataframe.columns:
            continue

        dtype = dataframe[feature_name].dtype

        if isinstance(dtype, pd.CategoricalDtype) or dtype == "object":
            categorical_features.append(feature_name)

    return categorical_features


def _read_reference_columns(path, columns):
    """
    Read selected columns from a parquet file.

    If column projection fails, the file is read and then subsetted.
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        dataframe = pd.read_parquet(path)
        available_columns = [column for column in columns if column in dataframe.columns]
        return dataframe[available_columns]


def load_categorical_reference_categories(reference_paths, categorical_features):
    """
    Build categorical reference categories from feature parquet files.

    Parameters
    ----------
    reference_paths : list of str or pathlib.Path
        Paths to feature parquet files used as category references.
    categorical_features : list of str
        Categorical feature names.

    Returns
    -------
    dict
        Categorical schema and metadata.
    """
    categorical_features = list(categorical_features)

    category_values = {
        feature_name: []
        for feature_name in categorical_features
    }

    used_reference_paths = []
    missing_reference_paths = []

    for reference_path in reference_paths:
        reference_path = Path(reference_path)

        if not reference_path.exists():
            missing_reference_paths.append(reference_path.as_posix())
            continue

        reference_data = _read_reference_columns(
            path=reference_path,
            columns=categorical_features,
        )

        if reference_data is None:
            continue

        used_reference_paths.append(reference_path.as_posix())

        for feature_name in categorical_features:
            if feature_name not in reference_data.columns:
                continue

            series = reference_data[feature_name]

            if isinstance(series.dtype, pd.CategoricalDtype):
                values = list(series.cat.categories)
            else:
                values = series.dropna().drop_duplicates().tolist()

            existing_values = set(category_values[feature_name])

            for value in values:
                if value not in existing_values:
                    category_values[feature_name].append(value)
                    existing_values.add(value)

    return {
        "categorical_features": categorical_features,
        "categories": category_values,
        "used_reference_paths": used_reference_paths,
        "missing_reference_paths": missing_reference_paths,
    }


def apply_categorical_schema(dataframe, categorical_schema):
    """
    Apply categorical dtype schema to an inference dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.
    categorical_schema : dict
        Categorical schema from load_categorical_reference_categories.

    Returns
    -------
    tuple
        Aligned dataframe and alignment report.
    """
    aligned_dataframe = dataframe.copy()
    report_records = []

    categories_by_feature = categorical_schema.get("categories", {})

    for feature_name, categories in categories_by_feature.items():
        if feature_name not in aligned_dataframe.columns:
            report_records.append(
                {
                    "feature_name": feature_name,
                    "exists": False,
                    "n_categories": len(categories),
                    "unknown_values": None,
                    "unknown_count": None,
                    "null_count_after_alignment": None,
                    "aligned": False,
                }
            )
            continue

        original_null_count = int(aligned_dataframe[feature_name].isna().sum())

        observed_values = (
            aligned_dataframe[feature_name]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        known_values = set(categories)
        unknown_values = [
            value
            for value in observed_values
            if value not in known_values
        ]

        dtype = pd.CategoricalDtype(categories=categories)
        aligned_dataframe[feature_name] = aligned_dataframe[feature_name].astype(dtype)

        null_count_after_alignment = int(aligned_dataframe[feature_name].isna().sum())
        unknown_count = max(0, null_count_after_alignment - original_null_count)

        report_records.append(
            {
                "feature_name": feature_name,
                "exists": True,
                "n_categories": len(categories),
                "unknown_values": unknown_values,
                "unknown_count": unknown_count,
                "null_count_after_alignment": null_count_after_alignment,
                "aligned": len(unknown_values) == 0,
            }
        )

    alignment_report = pd.DataFrame(report_records)

    return aligned_dataframe, alignment_report


def validate_categorical_alignment(alignment_report):
    """
    Validate categorical alignment results.

    Parameters
    ----------
    alignment_report : pandas.DataFrame
        Categorical alignment report.

    Returns
    -------
    pandas.DataFrame
        Validation checks.
    """
    checks = []

    if alignment_report.empty:
        return pd.DataFrame(
            [
                {
                    "check": "categorical_alignment_required",
                    "passed": True,
                    "critical": False,
                    "details": "No categorical model features were detected.",
                }
            ]
        )

    missing_features = alignment_report[
        alignment_report["exists"] == False
    ]["feature_name"].tolist()

    features_with_unknowns = alignment_report[
        alignment_report["unknown_count"].fillna(0) > 0
    ][["feature_name", "unknown_count"]].to_dict("records")

    checks.append(
        {
            "check": "categorical_features_exist",
            "passed": len(missing_features) == 0,
            "critical": True,
            "details": (
                "All categorical features exist."
                if len(missing_features) == 0
                else f"Missing categorical features: {missing_features}"
            ),
        }
    )

    checks.append(
        {
            "check": "no_unknown_categorical_values",
            "passed": len(features_with_unknowns) == 0,
            "critical": True,
            "details": (
                "No unknown categorical values after alignment."
                if len(features_with_unknowns) == 0
                else f"Features with unknown values: {features_with_unknowns}"
            ),
        }
    )

    return pd.DataFrame(checks)
