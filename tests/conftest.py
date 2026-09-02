import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def sample_employee_payload():
    row = pd.read_csv("data/processed/employees.csv").iloc[0].to_dict()
    payload = {}
    for key, value in row.items():
        payload[key] = value.item() if hasattr(value, "item") else value
    return payload


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
