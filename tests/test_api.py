from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict():
    payload = {
        "tenure": 12,
        "MonthlyCharges": 85.5,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["churn_prediction"] in ["Yes", "No"]
