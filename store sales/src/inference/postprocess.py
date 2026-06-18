"""
Prediction postprocessing utilities.
"""

from pathlib import Path

import pandas as pd


def build_prediction_output(
    source_dataframe,
    prediction_result,
    grain_columns,
    prediction_column="prediction",
    id_column="id",
    metadata=None,
):
    """
    Build the final prediction output table.

    Parameters
    ----------
    source_dataframe : pandas.DataFrame
        Original inference dataframe.
    prediction_result : pandas.DataFrame
        Prediction dataframe returned by the prediction module.
    grain_columns : list of str
        Columns defining prediction grain.
    prediction_column : str
        Final prediction column.
    id_column : str
        Optional identifier column.
    metadata : dict, optional
        Metadata columns to append to the output.

    Returns
    -------
    pandas.DataFrame
        Final prediction output.
    """
    if len(source_dataframe) != len(prediction_result):
        raise ValueError(
            "Source dataframe and prediction result must have the same number of rows. "
            f"Source rows: {len(source_dataframe)}; prediction rows: {len(prediction_result)}"
        )

    missing_grain_columns = sorted(set(grain_columns) - set(source_dataframe.columns))

    if missing_grain_columns:
        raise ValueError(
            f"Source dataframe is missing grain columns: {missing_grain_columns}"
        )

    output_columns = []

    if id_column in source_dataframe.columns:
        output_columns.append(id_column)

    output_columns.extend(grain_columns)

    prediction_output = source_dataframe[output_columns].reset_index(drop=True).copy()
    prediction_result = prediction_result.reset_index(drop=True).copy()

    for column in prediction_result.columns:
        if column in prediction_output.columns:
            raise ValueError(
                f"Prediction result column conflicts with output column: {column}"
            )

        prediction_output[column] = prediction_result[column]

    if metadata:
        for key, value in metadata.items():
            prediction_output[key] = value

    return prediction_output


def build_prediction_summary(
    prediction_output,
    prediction_column="prediction",
    clipped_prediction_column="prediction_clipped",
):
    """
    Build an aggregate summary for final predictions.

    Parameters
    ----------
    prediction_output : pandas.DataFrame
        Final prediction output.
    prediction_column : str
        Final prediction column.
    clipped_prediction_column : str
        Clipping flag column.

    Returns
    -------
    dict
        Prediction summary.
    """
    predictions = prediction_output[prediction_column]

    summary = {
        "rows": int(len(prediction_output)),
        "prediction_min": float(predictions.min()),
        "prediction_mean": float(predictions.mean()),
        "prediction_median": float(predictions.median()),
        "prediction_max": float(predictions.max()),
        "prediction_sum": float(predictions.sum()),
        "prediction_nulls": int(predictions.isna().sum()),
        "prediction_negatives": int((predictions < 0).sum()),
    }

    if clipped_prediction_column in prediction_output.columns:
        summary["clipped_predictions"] = int(
            prediction_output[clipped_prediction_column].sum()
        )

    if "date" in prediction_output.columns:
        summary["min_date"] = str(pd.to_datetime(prediction_output["date"]).min().date())
        summary["max_date"] = str(pd.to_datetime(prediction_output["date"]).max().date())
        summary["n_dates"] = int(pd.to_datetime(prediction_output["date"]).nunique())

    if "store_nbr" in prediction_output.columns:
        summary["n_stores"] = int(prediction_output["store_nbr"].nunique())

    if "family" in prediction_output.columns:
        summary["n_families"] = int(prediction_output["family"].nunique())

    return summary


def build_daily_prediction_summary(
    prediction_output,
    prediction_column="prediction",
    date_column="date",
):
    """
    Build daily prediction summary.

    Parameters
    ----------
    prediction_output : pandas.DataFrame
        Final prediction output.
    prediction_column : str
        Final prediction column.
    date_column : str
        Date column.

    Returns
    -------
    pandas.DataFrame
        Daily prediction summary.
    """
    if date_column not in prediction_output.columns:
        raise ValueError(f"Date column not found: {date_column}")

    daily_summary = (
        prediction_output
        .assign(**{date_column: pd.to_datetime(prediction_output[date_column])})
        .groupby(date_column, as_index=False)
        .agg(
            rows=(prediction_column, "size"),
            predicted_sales_sum=(prediction_column, "sum"),
            predicted_sales_mean=(prediction_column, "mean"),
            predicted_sales_median=(prediction_column, "median"),
            predicted_sales_min=(prediction_column, "min"),
            predicted_sales_max=(prediction_column, "max"),
        )
        .sort_values(date_column)
    )

    return daily_summary


def build_submission_file(
    prediction_output,
    sample_submission,
    id_column="id",
    prediction_column="prediction",
    submission_target_column="sales",
):
    """
    Build a Kaggle submission dataframe.

    Parameters
    ----------
    prediction_output : pandas.DataFrame
        Final prediction output.
    sample_submission : pandas.DataFrame
        Sample submission dataframe.
    id_column : str
        Submission identifier column.
    prediction_column : str
        Final prediction column.
    submission_target_column : str
        Kaggle target column.

    Returns
    -------
    pandas.DataFrame
        Submission dataframe.
    """
    if sample_submission is None:
        return None

    if id_column not in sample_submission.columns:
        raise ValueError(
            f"Sample submission is missing id column: {id_column}"
        )

    if id_column not in prediction_output.columns:
        raise ValueError(
            f"Prediction output is missing id column: {id_column}"
        )

    if prediction_column not in prediction_output.columns:
        raise ValueError(
            f"Prediction output is missing prediction column: {prediction_column}"
        )

    if prediction_output[id_column].duplicated().sum() > 0:
        raise ValueError("Prediction output contains duplicated ids.")

    submission = sample_submission[[id_column]].merge(
        prediction_output[[id_column, prediction_column]],
        on=id_column,
        how="left",
        validate="one_to_one",
    )

    missing_predictions = submission[prediction_column].isna().sum()

    if missing_predictions > 0:
        raise ValueError(
            f"Submission contains missing predictions: {missing_predictions}"
        )

    submission = submission.rename(
        columns={prediction_column: submission_target_column}
    )

    return submission[[id_column, submission_target_column]]
