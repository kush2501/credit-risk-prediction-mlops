import pandas as pd
import shap
import mlflow
import mlflow.xgboost
import dagshub

from src.utils.common import load_object
from src.services.prediction_service import PredictionService


# --------------------------------------------------
# 1. Initialize DagsHub + MLflow
# --------------------------------------------------

dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)

print("DagsHub + MLflow connected")


# --------------------------------------------------
# 2. Load preprocessor
# --------------------------------------------------

preprocessor = load_object(
    "artifacts/data_transformation/preprocessor.pkl"
)

print("Preprocessor loaded successfully")


# --------------------------------------------------
# 3. Load champion XGBoost model
# --------------------------------------------------

MODEL_URI = "models:/CreditRiskModel@champion"

xgb_model = mlflow.xgboost.load_model(
    MODEL_URI
)

print("Champion XGBoost model loaded successfully")


# --------------------------------------------------
# 4. Create SHAP explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(
    xgb_model
)

print("SHAP TreeExplainer created successfully")


# --------------------------------------------------
# 5. Create Prediction Service
# --------------------------------------------------

prediction_service = PredictionService(
    model=xgb_model,
    preprocessor=preprocessor,
    explainer=explainer,
)

print("PredictionService created successfully")


# --------------------------------------------------
# 6. Customer input
# --------------------------------------------------

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

print("\nCustomer input:")
print(customer_data)


# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------

prediction, probability = (
    prediction_service.predict(
        customer_data
    )
)

print("\nPrediction:")
print(prediction)

print("\nProbability:")
print(probability)


shap_values = prediction_service.explain(
    customer_data
)

print("\nSHAP values:")
print(shap_values.values)

print("\nSHAP values shape:")
print(shap_values.values.shape)

print("\nBase value:")
print(shap_values.base_values)