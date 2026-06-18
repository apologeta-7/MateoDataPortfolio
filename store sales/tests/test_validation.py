import pytest

from api.exceptions import ApiError


def test_predict_records_rejects_empty_request(forecast_service):
    with pytest.raises(ApiError) as exc_info:
        forecast_service.predict_records(records=[])

    assert exc_info.value.code == "EMPTY_REQUEST"


def test_predict_records_rejects_missing_model_features(
    forecast_service,
    sample_feature_records,
):
    record = dict(sample_feature_records[0])
    record.pop(forecast_service.feature_names[0])

    with pytest.raises(ApiError) as exc_info:
        forecast_service.predict_records(records=[record])

    assert exc_info.value.code == "MISSING_MODEL_FEATURES"


def test_predict_records_rejects_invalid_date(
    forecast_service,
    sample_feature_records,
):
    record = dict(sample_feature_records[0])
    record["date"] = "not-a-date"

    with pytest.raises(ApiError) as exc_info:
        forecast_service.predict_records(records=[record])

    assert exc_info.value.code == "INVALID_DATE"


def test_predict_records_rejects_duplicated_prediction_grain(
    forecast_service,
    sample_feature_records,
):
    duplicate_record = dict(sample_feature_records[0])

    with pytest.raises(ApiError) as exc_info:
        forecast_service.predict_records(records=[duplicate_record, duplicate_record])

    assert exc_info.value.code == "DUPLICATED_GRAIN"

