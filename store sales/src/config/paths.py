"""
Project path utilities.

This module centralizes path discovery and common project directories.
"""

from pathlib import Path


def find_project_root(start_path=None, markers=("data", "models", "notebooks")):
    """
    Find the project root by walking upward until expected project folders are found.

    Parameters
    ----------
    start_path : str or pathlib.Path, optional
        Starting location for the search. If None, the current working directory is used.
    markers : tuple of str
        Directory or file names expected at the project root.

    Returns
    -------
    pathlib.Path
        Resolved project root path.
    """
    current_path = Path(start_path or Path.cwd()).resolve()

    for candidate in [current_path, *current_path.parents]:
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise FileNotFoundError(
        "Project root could not be found. "
        f"Checked from: {current_path}"
    )


def get_project_paths(project_root=None):
    """
    Build the main project path registry.

    Parameters
    ----------
    project_root : str or pathlib.Path, optional
        Project root path. If None, it is detected automatically.

    Returns
    -------
    dict
        Dictionary with standard project directories.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()

    paths = {
        "project_root": root,
        "data_dir": root / "data",
        "raw_dir": root / "data" / "raw",
        "interim_dir": root / "data" / "interim",
        "processed_dir": root / "data" / "processed",
        "features_dir": root / "data" / "features",
        "predictions_dir": root / "data" / "predictions",
        "models_dir": root / "models",
        "lightgbm_models_dir": root / "models" / "lightgbm",
        "reports_dir": root / "reports",
        "reports_tables_dir": root / "reports" / "tables",
        "reports_inference_dir": root / "reports" / "inference",
        "mlruns_dir": root / "mlruns",
        "src_dir": root / "src",
    }

    return paths


def ensure_output_directories(paths):
    """
    Ensure expected output directories exist.

    Parameters
    ----------
    paths : dict
        Project path registry.

    Returns
    -------
    None
    """
    output_keys = [
        "predictions_dir",
        "reports_inference_dir",
    ]

    for key in output_keys:
        paths[key].mkdir(parents=True, exist_ok=True)


def relative_to_project(path, project_root=None):
    """
    Convert an absolute path to a project-relative string when possible.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to convert.
    project_root : str or pathlib.Path, optional
        Project root path. If None, it is detected automatically.

    Returns
    -------
    str
        Project-relative path when possible, otherwise absolute path.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()
    resolved_path = Path(path).resolve()

    try:
        return resolved_path.relative_to(root).as_posix()
    except ValueError:
        return resolved_path.as_posix()
