"""
Prediction utilities for LightGBM inference.
"""

import numpy as np
import pandas as pd


def get_prediction_iteration(model):
    """
    Return the best available prediction iteration for a LightGBM Booster.

    Parameters
    ----------
    model : lightgbm.Booster
        Loaded LightGBM Booster.

    Returns
    -------
    int or None
        Best iteration when available, otherwise None.
    """
    best_iteration = getattr(model, "best_iteration", None)

    if best_iteration is None or best_iteration <= 0:
        return None

    return best_iteration


def predict_log_sales(model, model_input):
    """
    Generate LightGBM predictions on the training target scale.

    The current model was trained on log1p(sales), so this function returns
    log-scale predictions.

    Parameters
    ----------
    model : lightgbm.Booster
        Loaded LightGBM Booster.
    model_input : pandas.DataFrame
        Model input matrix using the exact feature contract order.

    Returns
    -------
    numpy.ndarray
        Log-scale predictions.
    """
    prediction_iteration = get_prediction_iteration(model)

    predictions = model.predict(
        model_input,
        num_iteration=prediction_iteration,
    )

    return np.asarray(predictions)


def inverse_transform_predictions(log_predictions):
    """
    Apply inverse transformation from log1p scale to sales scale.

    Parameters
    ----------
    log_predictions : array-like
        Log-scale predictions.

    Returns
    -------
    numpy.ndarray
        Sales-scale predictions.
    """
    return np.expm1(log_predictions)


def clip_forecast_predictions(predictions, lower_bound=0.0):
    """
    Clip forecast predictions to a lower bound.

    Parameters
    ----------
    predictions : array-like
        Sales-scale predictions.
    lower_bound : float
        Minimum allowed prediction value.

    Returns
    -------
    numpy.ndarray
        Clipped predictions.
    """
    return np.clip(predictions, a_min=lower_bound, a_max=None)


def generate_predictions(
    model,
    model_input,
    prediction_column="prediction",
    log_prediction_column="prediction_log",
    raw_prediction_column="prediction_raw",
    clipped_prediction_column="prediction_clipped",
    lower_bound=0.0,
):
    """
    Generate final sales predictions from a LightGBM Booster.

    Parameters
    ----------
    model : lightgbm.Booster
        Loaded LightGBM Booster.
    model_input : pandas.DataFrame
        Model input matrix.
    prediction_column : str
        Final prediction column name.
    log_prediction_column : str
        Log-scale prediction column name.
    raw_prediction_column : str
        Sales-scale prediction before clipping.
    clipped_prediction_column : str
        Boolean flag indicating whether clipping changed the prediction.
    lower_bound : float
        Minimum allowed prediction value.

    Returns
    -------
    pandas.DataFrame
        Prediction dataframe.
    """
    log_predictions = predict_log_sales(
        model=model,
        model_input=model_input,
    )

    raw_predictions = inverse_transform_predictions(log_predictions)
    final_predictions = clip_forecast_predictions(
        predictions=raw_predictions,
        lower_bound=lower_bound,
    )

    prediction_result = pd.DataFrame(
        {
            log_prediction_column: log_predictions,
            raw_prediction_column: raw_predictions,
            prediction_column: final_predictions,
        }
    )

    prediction_result[clipped_prediction_column] = (
        prediction_result[prediction_column]
        != prediction_result[raw_prediction_column]
    )

    return prediction_result


def build_prediction_diagnostics(
    prediction_result,
    prediction_column="prediction",
    raw_prediction_column="prediction_raw",
    clipped_prediction_column="prediction_clipped",
):
    """
    Build basic diagnostics for generated predictions.

    Parameters
    ----------
    prediction_result : pandas.DataFrame
        Prediction dataframe.
    prediction_column : str
        Final prediction column name.
    raw_prediction_column : str
        Raw prediction column name.
    clipped_prediction_column : str
        Clipping flag column name.

    Returns
    -------
    dict
        Prediction diagnostics.
    """
    predictions = prediction_result[prediction_column]

    if clipped_prediction_column in prediction_result.columns:
        clipped_predictions = int(prediction_result[clipped_prediction_column].sum())
    else:
        clipped_predictions = None

    raw_negative_predictions = None
    if raw_prediction_column in prediction_result.columns:
        raw_negative_predictions = int(
            (prediction_result[raw_prediction_column] < 0).sum()
        )

    return {
        "rows": int(len(prediction_result)),
        "prediction_min": float(predictions.min()),
        "prediction_mean": float(predictions.mean()),
        "prediction_median": float(predictions.median()),
        "prediction_max": float(predictions.max()),
        "prediction_sum": float(predictions.sum()),
        "raw_negative_predictions": raw_negative_predictions,
        "clipped_predictions": clipped_predictions,
    }
