"""
Generic I/O utilities.
"""

import json
from pathlib import Path

import pandas as pd


def read_json(path):
    """
    Read a JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON file path.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path, indent=2):
    """
    Save a dictionary as JSON.

    Parameters
    ----------
    data : dict
        Data to save.
    path : str or pathlib.Path
        Output JSON path.
    indent : int
        JSON indentation.

    Returns
    -------
    pathlib.Path
        Saved file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    return path


def read_parquet(path):
    """
    Read a parquet file.

    Parameters
    ----------
    path : str or pathlib.Path
        Parquet file path.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """
    return pd.read_parquet(Path(path))


def save_parquet(dataframe, path, index=False):
    """
    Save a dataframe as parquet.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to save.
    path : str or pathlib.Path
        Output parquet path.
    index : bool
        Whether to save the index.

    Returns
    -------
    pathlib.Path
        Saved file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=index)
    return path


def read_csv_if_exists(path, **kwargs):
    """
    Read a CSV file only if it exists.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file path.
    **kwargs
        Additional pandas read_csv arguments.

    Returns
    -------
    pandas.DataFrame or None
        Loaded dataframe, or None if the file does not exist.
    """
    path = Path(path)

    if not path.exists():
        return None

    return pd.read_csv(path, **kwargs)


def save_csv(dataframe, path, index=False):
    """
    Save a dataframe as CSV.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to save.
    path : str or pathlib.Path
        Output CSV path.
    index : bool
        Whether to save the index.

    Returns
    -------
    pathlib.Path
        Saved file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=index)
    return path


def file_size_mb(path):
    """
    Return file size in MB.

    Parameters
    ----------
    path : str or pathlib.Path
        File path.

    Returns
    -------
    float or None
        File size in MB, or None when the file does not exist.
    """
    path = Path(path)

    if not path.exists():
        return None

    return round(path.stat().st_size / (1024 ** 2), 4)


def dataframe_memory_mb(dataframe):
    """
    Return dataframe memory usage in MB.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    float
        Memory usage in MB.
    """
    return round(dataframe.memory_usage(deep=True).sum() / (1024 ** 2), 4)
