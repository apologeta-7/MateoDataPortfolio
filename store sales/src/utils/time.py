"""
Time utilities.
"""

from datetime import datetime, timezone


def utc_now_iso():
    """
    Return the current UTC timestamp in ISO format.

    Returns
    -------
    str
        UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()
