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

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


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

FRONTEND_DIR = Path(__file__).parent / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


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

        explanations = prediction_service.explain(
            customer_data
        )

        return CustomerPredictionResponse(
            prediction=int(prediction[0]),
            non_default_probability=float(probability[0][0]),
            default_probability=float(probability[0][1]),
            explanations=explanations,
        )
    except Exception as e:
        logger.error(
            f"Prediction failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be generated."
        )