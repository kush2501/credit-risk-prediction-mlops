import pandas as pd
import shap
import mlflow
import mlflow.xgboost
import dagshub

from fastapi import FastAPI, HTTPException

from src.logger.logger import logger

from src.utils.common import load_object
from src.services.prediction_service import PredictionService
from src.api.schemas import (
    CustomerPredictionRequest,
    CustomerPredictionResponse,
)


# --------------------------------------------------
# DagsHub + MLflow
# --------------------------------------------------

dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)


# --------------------------------------------------
# Load Preprocessor
# --------------------------------------------------

preprocessor = load_object(
    "artifacts/data_transformation/preprocessor.pkl"
)


# --------------------------------------------------
# Load Champion Model
# --------------------------------------------------

MODEL_URI = "models:/CreditRiskModel@champion"

xgb_model = mlflow.xgboost.load_model(
    MODEL_URI
)


# --------------------------------------------------
# Create SHAP Explainer
# --------------------------------------------------

explainer = shap.TreeExplainer(
    xgb_model
)


# --------------------------------------------------
# Create Prediction Service
# --------------------------------------------------

prediction_service = PredictionService(
    model=xgb_model,
    preprocessor=preprocessor,
    explainer=explainer,
)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production API for credit risk prediction",
    version="1.0.0",
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=CustomerPredictionResponse,
)
def predict(
    customer: CustomerPredictionRequest
):
    try:
        customer_data = pd.DataFrame(
            [customer.model_dump()]
        )

        prediction, probability = (
            prediction_service.predict(
                customer_data
            )
        )

        return {
            "prediction": int(prediction[0]),
            "non_default_probability": float(
                probability[0][0]
            ),
            "default_probability": float(
                probability[0][1]
            ),
        }

    except Exception as e:
        logger.error(
            f"Prediction failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be generated."
        )