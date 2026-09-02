import pytest
from pydantic import ValidationError

from app.validation.schemas import AttritionPredictRequest, EngagementScoreRequest


def test_missing_required_column_caught(sample_employee_payload):
    payload = dict(sample_employee_payload)
    payload.pop("MonthlyIncome")
    with pytest.raises(ValidationError) as exc_info:
        AttritionPredictRequest.model_validate(payload)
    assert "MonthlyIncome" in str(exc_info.value)


def test_invalid_engagement_score_rejected():
    with pytest.raises(ValidationError):
        EngagementScoreRequest.model_validate({"Engagement Score": 9})
    with pytest.raises(ValidationError):
        EngagementScoreRequest.model_validate({"Engagement Score": 0})


def test_invalid_engagement_score_rejected_on_attrition_payload(sample_employee_payload):
    payload = dict(sample_employee_payload)
    payload["Engagement Score"] = 9
    with pytest.raises(ValidationError) as exc_info:
        AttritionPredictRequest.model_validate(payload)
    assert "Engagement Score" in str(exc_info.value)
