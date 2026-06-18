import math


def test_forecast_service_readiness_is_ready(forecast_service):
    readiness = forecast_service.readiness()

    assert readiness["ready"] is True
    assert readiness["model_loaded"] is True
    assert len(readiness["artifact_checks"]) >= 4


def test_forecast_service_predicts_model_ready_records(
    forecast_service,
    sample_feature_records,
):
    response = forecast_service.predict_records(
        records=sample_feature_records,
        include_diagnostics=True,
    )

    assert response["rows"] == len(sample_feature_records)
    assert response["summary"]["rows"] == len(sample_feature_records)
    assert response["diagnostics"] is not None
    assert response["summary"]["prediction_nulls"] == 0
    assert response["summary"]["prediction_negatives"] == 0

    for prediction in response["predictions"]:
        value = prediction["prediction"]
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value >= 0

