from app.ml.predictor import get_risk_bucket, predict_attrition_probability


def test_attrition_prediction_returns_real_probability(sample_employee_payload):
    probability = predict_attrition_probability(sample_employee_payload)
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_risk_level_assigned_correctly():
    assert get_risk_bucket(0.0) == "Low"
    assert get_risk_bucket(0.3999) == "Low"
    assert get_risk_bucket(0.40) == "Medium"
    assert get_risk_bucket(0.69) == "Medium"
    assert get_risk_bucket(0.70) == "High"
    assert get_risk_bucket(1.0) == "High"
