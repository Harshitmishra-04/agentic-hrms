def test_predict_attrition_success(client, sample_employee_payload):
    response = client.post("/predict/attrition", json=sample_employee_payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["attrition_probability"] <= 1.0
    assert body["risk_bucket"] in {"Low", "Medium", "High"}


def test_predict_attrition_missing_column_returns_400(client, sample_employee_payload):
    payload = dict(sample_employee_payload)
    payload.pop("MonthlyIncome")
    response = client.post("/predict/attrition", json=payload)
    assert response.status_code == 400


def test_predict_attrition_invalid_engagement_returns_400(client, sample_employee_payload):
    payload = dict(sample_employee_payload)
    payload["Engagement Score"] = 9
    response = client.post("/predict/attrition", json=payload)
    assert response.status_code == 400


def test_dashboard_endpoints_return_200(client):
    assert client.get("/dashboard/summary").status_code == 200
    assert client.get("/dashboard/attrition-by-department").status_code == 200
    assert client.get("/dashboard/skill-gaps").status_code == 200
    assert client.get("/dashboard/recommendations").status_code == 200


def test_skill_gaps_invalid_limit_returns_400(client):
    assert client.get("/dashboard/skill-gaps?limit=0").status_code == 400


def test_employee_lookup_status_codes(client):
    assert client.get("/employees/1").status_code == 200
    assert client.get("/employees/0").status_code == 400
    assert client.get("/employees/999999").status_code == 404
