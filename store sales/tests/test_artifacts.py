from src.config.inference import get_inference_artifact_paths
from src.config.paths import get_project_paths
from src.inference.artifacts import load_feature_contract
from src.inference.model import validate_model_against_feature_contract


def test_required_serving_artifacts_exist(project_root):
    paths = get_project_paths(project_root=project_root)
    artifact_paths = get_inference_artifact_paths(paths)

    required_keys = [
        "baseline_test_features",
        "lightgbm_model",
        "lightgbm_config",
        "lightgbm_feature_list",
        "lightgbm_full_model",
        "lightgbm_full_config",
        "final_model_recommendation",
    ]

    missing = [
        artifact_key
        for artifact_key in required_keys
        if not artifact_paths[artifact_key].exists()
    ]

    assert missing == []


def test_model_feature_contract_matches_loaded_model(forecast_service, project_root):
    paths = get_project_paths(project_root=project_root)
    artifact_paths = get_inference_artifact_paths(paths)
    feature_contract = load_feature_contract(artifact_paths["lightgbm_feature_list"])

    assert feature_contract["n_features"] == 41
    assert forecast_service.model_metadata["num_features"] == 41

    validation = validate_model_against_feature_contract(
        model=forecast_service.model,
        feature_names=feature_contract["feature_names"],
    )

    failed_critical = validation[
        (validation["critical"] == True) & (validation["passed"] == False)
    ]
    assert failed_critical.empty

