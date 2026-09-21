from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_predict_endpoint():

    payload = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.20,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "non_default_probability" in result
    assert "default_probability" in result

    assert result["prediction"] in [0, 1]

    assert 0 <= result["non_default_probability"] <= 1
    assert 0 <= result["default_probability"] <= 1


def test_predict_invalid_loan_grade():

    payload = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3,
        "loan_intent": "EDUCATION",
        "loan_grade": "Z",
        "loan_amnt": 10000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.20,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422