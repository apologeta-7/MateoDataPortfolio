"""
model.py
--------
Reusable model evaluation and training functions for the House Prices project.
Author: Mateo Pascual
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from lightgbm import LGBMRegressor


def evaluate_model(preprocessor, X, y, random_state=42):
    """
    Evaluates a model (LightGBM by default) using cross-validation RMSE.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline built from `prep.py`.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values (log-transformed).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Cross-validated RMSE (lower is better).
    """
    model = LGBMRegressor(random_state=random_state)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    scores = cross_val_score(pipe, X, y, scoring=scorer, cv=cv)
    rmse = np.sqrt(-scores.mean())

    print(f"Cross-validated RMSE_log: {rmse:.4f}")
    return rmse
