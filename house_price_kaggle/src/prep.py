"""
prep.py
--------
Reusable preprocessing utilities for the House Prices project.
Author: Mateo Pascual
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that handles
    both numeric and categorical preprocessing:
      - Numeric: median imputation + StandardScaler
      - Categorical: most frequent imputation + OneHotEncoder

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe used to infer column types.

    Returns
    -------
    ColumnTransformer
        Configured preprocessor ready for use in a Pipeline.
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    return preprocessor
