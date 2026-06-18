def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_endpoint(client):
    response = client.get("/health/ready")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_feature_contract_endpoint(client):
    response = client.get("/api/v1/features")
    body = response.json()

    assert response.status_code == 200
    assert body["n_features"] == 41
    assert body["prediction_grain"] == ["date", "store_nbr", "family"]
    assert "family" in body["feature_names"]


def test_model_metadata_endpoint(client):
    response = client.get("/api/v1/model")
    body = response.json()

    assert response.status_code == 200
    assert body["recommended_model_version"] == "lightgbm_v1"
    assert body["serving_model_version"] in {"lightgbm_v1", "lightgbm_v1_full"}
    assert body["n_features"] == 41


def test_predict_endpoint(client, sample_feature_records):
    response = client.post(
        "/api/v1/predict",
        json={"records": sample_feature_records[:2], "include_diagnostics": False},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["rows"] == 2
    assert body["summary"]["rows"] == 2
    assert body["summary"]["prediction_negatives"] == 0

