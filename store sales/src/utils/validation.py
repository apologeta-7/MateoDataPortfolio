"""
Generic validation utilities.
"""

import pandas as pd


def require_columns(dataframe, required_columns, dataframe_name="dataframe"):
    """
    Validate that a dataframe contains required columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe to validate.
    required_columns : list of str
        Required column names.
    dataframe_name : str
        Human-readable dataframe name.

    Returns
    -------
    list
        Missing columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing_columns}"
        )

    return missing_columns


def validate_no_critical_failures(validation_table):
    """
    Raise an error if a validation table contains failed critical checks.

    Parameters
    ----------
    validation_table : pandas.DataFrame
        Validation table with columns: check, passed, critical, details.

    Returns
    -------
    bool
        True when no critical failures exist.

    Raises
    ------
    ValueError
        If at least one critical check failed.
    """
    required_columns = ["check", "passed", "critical", "details"]
    missing_columns = sorted(set(required_columns) - set(validation_table.columns))

    if missing_columns:
        raise ValueError(
            f"Validation table is missing columns: {missing_columns}"
        )

    failed_critical_checks = validation_table[
        (validation_table["critical"] == True)
        & (validation_table["passed"] == False)
    ]

    if not failed_critical_checks.empty:
        details = "\n".join(
            [
                f"- {row.check}: {row.details}"
                for row in failed_critical_checks.itertuples(index=False)
            ]
        )

        raise ValueError(f"Critical validation checks failed:\n{details}")

    return True


def build_check(check, passed, critical, details):
    """
    Build a standard validation check record.

    Parameters
    ----------
    check : str
        Check name.
    passed : bool
        Whether the check passed.
    critical : bool
        Whether the check is critical.
    details : str
        Human-readable details.

    Returns
    -------
    dict
        Standard validation record.
    """
    return {
        "check": check,
        "passed": bool(passed),
        "critical": bool(critical),
        "details": details,
    }


def checks_to_dataframe(checks):
    """
    Convert validation records to a dataframe.

    Parameters
    ----------
    checks : list of dict
        Validation records.

    Returns
    -------
    pandas.DataFrame
        Validation dataframe.
    """
    return pd.DataFrame(checks)
