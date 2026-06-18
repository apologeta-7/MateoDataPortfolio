"""
Inference configuration.

This module centralizes model serving constants and artifact paths.
"""

FINAL_RECOMMENDED_MODEL_VERSION = "lightgbm_v1"
FULL_SERVING_MODEL_VERSION = "lightgbm_v1_full"

PREDICTION_GRAIN = ["date", "store_nbr", "family"]

TARGET_COLUMN = "sales"
PREDICTION_COLUMN = "prediction"

EXPECTED_FORECAST_HORIZON_DAYS = 16

INFERENCE_OUTPUT_VERSION = "final_model_inference_v1"
SRC_INFERENCE_OUTPUT_VERSION = "final_model_inference_v1_src"

MLFLOW_EXPERIMENT_NAME = "store_sales_forecasting"


def get_inference_artifact_paths(paths):
    """
    Build the inference artifact path registry.

    Parameters
    ----------
    paths : dict
        Project path registry.

    Returns
    -------
    dict
        Dictionary with inference artifact paths.
    """
    lightgbm_models_dir = paths["lightgbm_models_dir"]
    reports_tables_dir = paths["reports_tables_dir"]
    reports_inference_dir = paths["reports_inference_dir"]
    predictions_dir = paths["predictions_dir"]
    features_dir = paths["features_dir"]
    raw_dir = paths["raw_dir"]

    return {
        "baseline_train_features": features_dir / "baseline_train_features.parquet",
        "baseline_valid_features": features_dir / "baseline_valid_features.parquet",
        "baseline_test_features": features_dir / "baseline_test_features.parquet",

        "sample_submission": raw_dir / "sample_submission.csv",

        "lightgbm_model": lightgbm_models_dir / "lightgbm_v1_model.txt",
        "lightgbm_config": lightgbm_models_dir / "lightgbm_v1_config.json",
        "lightgbm_feature_list": lightgbm_models_dir / "lightgbm_v1_feature_list.json",
        "lightgbm_experiment_summary": lightgbm_models_dir / "lightgbm_v1_experiment_summary.json",

        "lightgbm_full_model": lightgbm_models_dir / "lightgbm_v1_full_model.txt",
        "lightgbm_full_config": lightgbm_models_dir / "lightgbm_v1_full_config.json",

        "final_model_recommendation": reports_tables_dir / "final_model_recommendation.csv",
        "mlflow_final_model_tracking_summary": reports_tables_dir / "mlflow_final_model_tracking_summary.csv",
        "mlflow_run_summary": reports_tables_dir / "mlflow_run_summary.csv",
        "mlflow_registered_artifacts": reports_tables_dir / "mlflow_registered_artifacts.csv",

        "notebook_08_predictions": predictions_dir / "final_model_inference_v1_predictions.parquet",
        "notebook_08_submission": predictions_dir / "final_model_inference_v1_submission.csv",

        "src_predictions": predictions_dir / "final_model_inference_v1_src_predictions.parquet",
        "src_submission": predictions_dir / "final_model_inference_v1_src_submission.csv",

        "src_report": reports_inference_dir / "final_model_inference_v1_src_report.json",
        "src_feature_validation": reports_inference_dir / "final_model_inference_v1_src_feature_validation.csv",
        "src_output_validation": reports_inference_dir / "final_model_inference_v1_src_output_validation.csv",
        "src_comparison": reports_inference_dir / "final_model_inference_v1_src_comparison.csv",
        "src_closure_criteria": reports_inference_dir / "final_model_inference_v1_src_closure_criteria.csv",
        "src_api_contract": reports_inference_dir / "final_model_inference_v1_src_api_contract.json",
    }
