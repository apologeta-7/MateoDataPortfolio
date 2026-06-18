"""
Inference-specific validation utilities.
"""

import numpy as np
import pandas as pd

from src.utils.validation import build_check, checks_to_dataframe


def validate_raw_inference_schema(
    dataframe,
    grain_columns,
    expected_forecast_horizon_days=16,
    target_column="sales",
    id_column="id",
    promotion_column="onpromotion",
):
    """
    Validate the raw batch inference schema.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Inference dataframe.
    grain_columns : list of str
        Columns defining the prediction grain.
    expected_forecast_horizon_days : int
        Expected number of forecast dates.
    target_column : str
        Target column that should not be present in inference input.
    id_column : str
        Optional submission identifier column.
    promotion_column : str
        Promotion column expected in the inference table.

    Returns
    -------
    pandas.DataFrame
        Validation table.
    """
    checks = []

    checks.append(
        build_check(
            check="input_not_empty",
            passed=len(dataframe) > 0,
            critical=True,
            details=f"Rows: {len(dataframe)}",
        )
    )

    missing_grain_columns = sorted(set(grain_columns) - set(dataframe.columns))
    checks.append(
        build_check(
            check="prediction_grain_columns_available",
            passed=len(missing_grain_columns) == 0,
            critical=True,
            details=(
                "All prediction grain columns are available."
                if len(missing_grain_columns) == 0
                else f"Missing grain columns: {missing_grain_columns}"
            ),
        )
    )

    checks.append(
        build_check(
            check="target_column_absent",
            passed=target_column not in dataframe.columns,
            critical=True,
            details=(
                f"Target column '{target_column}' is absent."
                if target_column not in dataframe.columns
                else f"Target column '{target_column}' should not be present in inference input."
            ),
        )
    )

    if len(missing_grain_columns) == 0:
        duplicated_grain_rows = dataframe.duplicated(subset=grain_columns).sum()
        checks.append(
            build_check(
                check="prediction_grain_is_unique",
                passed=duplicated_grain_rows == 0,
                critical=True,
                details=f"Duplicated grain rows: {duplicated_grain_rows}",
            )
        )

        if "date" in grain_columns:
            n_dates = dataframe["date"].nunique()
            checks.append(
                build_check(
                    check="expected_forecast_horizon",
                    passed=n_dates == expected_forecast_horizon_days,
                    critical=True,
                    details=(
                        f"Unique dates: {n_dates}; "
                        f"expected: {expected_forecast_horizon_days}"
                    ),
                )
            )

        if {"date", "store_nbr", "family"}.issubset(dataframe.columns):
            n_dates = dataframe["date"].nunique()
            n_stores = dataframe["store_nbr"].nunique()
            n_families = dataframe["family"].nunique()
            expected_rows = n_dates * n_stores * n_families

            checks.append(
                build_check(
                    check="complete_store_family_date_grid",
                    passed=len(dataframe) == expected_rows,
                    critical=True,
                    details=(
                        f"Rows: {len(dataframe)}; expected grid rows: {expected_rows}; "
                        f"dates: {n_dates}; stores: {n_stores}; families: {n_families}"
                    ),
                )
            )

    if id_column in dataframe.columns:
        duplicated_ids = dataframe[id_column].duplicated().sum()
        checks.append(
            build_check(
                check="id_column_is_unique",
                passed=duplicated_ids == 0,
                critical=True,
                details=f"Duplicated ids: {duplicated_ids}",
            )
        )
    else:
        checks.append(
            build_check(
                check="id_column_available",
                passed=False,
                critical=False,
                details=f"Optional id column '{id_column}' is not available.",
            )
        )

    if promotion_column in dataframe.columns:
        negative_promotions = (dataframe[promotion_column] < 0).sum()
        null_promotions = dataframe[promotion_column].isna().sum()

        checks.append(
            build_check(
                check="promotion_values_are_non_negative",
                passed=negative_promotions == 0,
                critical=True,
                details=f"Negative promotion values: {negative_promotions}",
            )
        )

        checks.append(
            build_check(
                check="promotion_values_not_null",
                passed=null_promotions == 0,
                critical=True,
                details=f"Null promotion values: {null_promotions}",
            )
        )
    else:
        checks.append(
            build_check(
                check="promotion_column_available",
                passed=False,
                critical=False,
                details=f"Optional promotion column '{promotion_column}' is not available.",
            )
        )

    return checks_to_dataframe(checks)


def build_feature_inventory(dataframe, feature_names):
    """
    Build a feature inventory against the model feature contract.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Inference dataframe.
    feature_names : list of str
        Expected model feature names.

    Returns
    -------
    pandas.DataFrame
        Feature inventory table.
    """
    records = []

    for position, feature_name in enumerate(feature_names):
        exists = feature_name in dataframe.columns

        if exists:
            series = dataframe[feature_name]
            dtype = str(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = float(series.isna().mean())
            unique_values = int(series.nunique(dropna=True))
        else:
            dtype = None
            null_count = None
            null_pct = None
            unique_values = None

        records.append(
            {
                "feature_position": position,
                "feature_name": feature_name,
                "exists": exists,
                "dtype": dtype,
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_values": unique_values,
            }
        )

    return pd.DataFrame(records)


def validate_feature_contract(dataframe, feature_names):
    """
    Validate that inference features satisfy the model feature contract.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Inference dataframe.
    feature_names : list of str
        Expected model feature names.

    Returns
    -------
    pandas.DataFrame
        Validation table.
    """
    checks = []

    missing_features = sorted(set(feature_names) - set(dataframe.columns))
    extra_columns = sorted(set(dataframe.columns) - set(feature_names))

    checks.append(
        build_check(
            check="all_model_features_available",
            passed=len(missing_features) == 0,
            critical=True,
            details=(
                "All model features are available."
                if len(missing_features) == 0
                else f"Missing features: {missing_features}"
            ),
        )
    )

    if len(missing_features) == 0:
        feature_matrix = dataframe[feature_names]
        null_counts = feature_matrix.isna().sum()
        features_with_nulls = null_counts[null_counts > 0].to_dict()

        checks.append(
            build_check(
                check="model_features_have_no_nulls",
                passed=len(features_with_nulls) == 0,
                critical=True,
                details=(
                    "No null values found in model features."
                    if len(features_with_nulls) == 0
                    else f"Features with nulls: {features_with_nulls}"
                ),
            )
        )

        feature_order_matches = list(feature_matrix.columns) == list(feature_names)

        checks.append(
            build_check(
                check="feature_order_matches_contract",
                passed=feature_order_matches,
                critical=True,
                details=(
                    f"Feature count: {len(feature_names)}"
                    if feature_order_matches
                    else "Feature order does not match the contract."
                ),
            )
        )

        unsupported_object_features = [
            column
            for column in feature_matrix.columns
            if str(feature_matrix[column].dtype) == "object"
        ]

        checks.append(
            build_check(
                check="no_raw_object_model_features",
                passed=len(unsupported_object_features) == 0,
                critical=False,
                details=(
                    "No raw object dtype model features found."
                    if len(unsupported_object_features) == 0
                    else f"Object dtype features: {unsupported_object_features}"
                ),
            )
        )

        finite_numeric_check = True
        non_finite_columns = []

        numeric_features = feature_matrix.select_dtypes(include=[np.number])

        if numeric_features.shape[1] > 0:
            finite_by_column = np.isfinite(numeric_features).all(axis=0)
            non_finite_columns = finite_by_column[
                finite_by_column == False
            ].index.tolist()
            finite_numeric_check = len(non_finite_columns) == 0

        checks.append(
            build_check(
                check="numeric_features_are_finite",
                passed=finite_numeric_check,
                critical=True,
                details=(
                    "All numeric model features are finite."
                    if finite_numeric_check
                    else f"Non-finite numeric features: {non_finite_columns}"
                ),
            )
        )

    checks.append(
        build_check(
            check="extra_columns_allowed",
            passed=True,
            critical=False,
            details=(
                f"Extra non-model columns available: {len(extra_columns)}. "
                "They will not be passed to the model."
            ),
        )
    )

    return checks_to_dataframe(checks)


def make_model_input_matrix(dataframe, feature_names):
    """
    Create the model input matrix using the exact feature contract order.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Inference dataframe.
    feature_names : list of str
        Expected model feature names.

    Returns
    -------
    pandas.DataFrame
        Model input matrix.
    """
    missing_features = sorted(set(feature_names) - set(dataframe.columns))

    if missing_features:
        raise ValueError(
            f"Cannot build model input matrix. Missing features: {missing_features}"
        )

    return dataframe.loc[:, feature_names].copy()


def validate_prediction_output(
    dataframe,
    prediction_column="prediction",
    grain_columns=None,
):
    """
    Validate final prediction output.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Prediction output dataframe.
    prediction_column : str
        Prediction column name.
    grain_columns : list of str, optional
        Columns defining prediction grain.

    Returns
    -------
    pandas.DataFrame
        Validation table.
    """
    checks = []

    checks.append(
        build_check(
            check="prediction_column_available",
            passed=prediction_column in dataframe.columns,
            critical=True,
            details=f"Prediction column: {prediction_column}",
        )
    )

    if prediction_column in dataframe.columns:
        predictions = dataframe[prediction_column]

        checks.append(
            build_check(
                check="predictions_not_null",
                passed=predictions.isna().sum() == 0,
                critical=True,
                details=f"Null predictions: {int(predictions.isna().sum())}",
            )
        )

        checks.append(
            build_check(
                check="predictions_are_non_negative",
                passed=(predictions < 0).sum() == 0,
                critical=True,
                details=f"Negative predictions: {int((predictions < 0).sum())}",
            )
        )

        finite_predictions = np.isfinite(predictions).all()

        checks.append(
            build_check(
                check="predictions_are_finite",
                passed=finite_predictions,
                critical=True,
                details="All predictions are finite." if finite_predictions else "Non-finite predictions found.",
            )
        )

    if grain_columns is not None:
        missing_grain_columns = sorted(set(grain_columns) - set(dataframe.columns))

        checks.append(
            build_check(
                check="prediction_output_grain_available",
                passed=len(missing_grain_columns) == 0,
                critical=True,
                details=(
                    "Prediction grain columns are available."
                    if len(missing_grain_columns) == 0
                    else f"Missing grain columns: {missing_grain_columns}"
                ),
            )
        )

        if len(missing_grain_columns) == 0:
            duplicated_grain_rows = dataframe.duplicated(subset=grain_columns).sum()

            checks.append(
                build_check(
                    check="prediction_output_grain_is_unique",
                    passed=duplicated_grain_rows == 0,
                    critical=True,
                    details=f"Duplicated output grain rows: {duplicated_grain_rows}",
                )
            )

    return checks_to_dataframe(checks)
