import pandas as pd
import shap
import mlflow
import mlflow.xgboost
import dagshub

from src.utils.common import load_object
from src.services.prediction_service import PredictionService


def test_prediction_service():

    # Initialize DagsHub + MLflow
    dagshub.init(
        repo_owner="kush2501",
        repo_name="credit-risk-prediction-mlops",
        mlflow=True,
    )

    # Load preprocessor
    preprocessor = load_object(
        "artifacts/data_transformation/preprocessor.pkl"
    )

    # Load champion XGBoost model
    model_uri = "models:/CreditRiskModel@champion"

    xgb_model = mlflow.xgboost.load_model(
        model_uri
    )

    # Create SHAP explainer
    explainer = shap.TreeExplainer(
        xgb_model
    )

    # Create PredictionService
    prediction_service = PredictionService(
        model=xgb_model,
        preprocessor=preprocessor,
        explainer=explainer,
    )

    # Customer input
    customer_data = pd.DataFrame([{
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
    }])

    # Test prediction
    prediction, probability = (
        prediction_service.predict(customer_data)
    )

    # Prediction checks
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]

    # Probability checks
    assert probability.shape == (1, 2)
    assert 0 <= probability[0][0] <= 1
    assert 0 <= probability[0][1] <= 1

    # Test SHAP explanation
    shap_values = prediction_service.explain(
        customer_data
    )

    # SHAP checks
    assert shap_values.values.shape == (1, 19)
    assert shap_values.base_values.shape == (1,)