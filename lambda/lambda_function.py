import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3
import joblib
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BUCKET_NAME = "credit-risk-prediction-mlops-522257406101"

MODEL_KEY = "model/best_model.pkl"
PREPROCESSOR_KEY = "model/preprocessor.pkl"

TABLE_NAME = "credit-risk-predictions"

MODEL_PATH = os.path.join(
    tempfile.gettempdir(),
    "best_model.pkl",
)

PREPROCESSOR_PATH = os.path.join(
    tempfile.gettempdir(),
    "preprocessor.pkl",
)


# ============================================================
# AWS clients
# ============================================================

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

table = dynamodb.Table(TABLE_NAME)


# ============================================================
# Load model and preprocessor
# ============================================================

def load_artifacts():
    """
    Download model and preprocessor from S3 only if
    they are not already available in Lambda /tmp.
    """

    if not os.path.exists(MODEL_PATH):
        s3.download_file(
            BUCKET_NAME,
            MODEL_KEY,
            MODEL_PATH,
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        s3.download_file(
            BUCKET_NAME,
            PREPROCESSOR_KEY,
            PREPROCESSOR_PATH,
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


# ============================================================
# Lambda Handler
# ============================================================

def lambda_handler(event, context):

    try:
        # ----------------------------------------------------
        # 1. Load model and preprocessing pipeline
        # ----------------------------------------------------

        model, preprocessor = load_artifacts()


        # ----------------------------------------------------
        # 2. Read request body
        # ----------------------------------------------------

        if isinstance(event, dict) and "body" in event:

            body = event["body"]

            if isinstance(body, str):
                customer_data = json.loads(body)
            else:
                customer_data = body

        else:
            customer_data = event


        # ----------------------------------------------------
        # 3. Convert input into DataFrame
        # ----------------------------------------------------

        customer_df = pd.DataFrame([customer_data])


        # ----------------------------------------------------
        # 4. Apply preprocessing
        # ----------------------------------------------------

        transformed_data = preprocessor.transform(
            customer_df
        )


        # ----------------------------------------------------
        # 5. Make prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(transformed_data)[0]
        )


        # ----------------------------------------------------
        # 6. Get prediction probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            transformed_data
        )[0]

        non_default_probability = float(
            probabilities[0]
        )

        default_probability = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # 7. Create prediction ID
        # ----------------------------------------------------

        prediction_id = str(uuid.uuid4())


        # ----------------------------------------------------
        # 8. Create timestamp
        # ----------------------------------------------------

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()


        # ----------------------------------------------------
        # 9. Save prediction to DynamoDB
        # ----------------------------------------------------

        table.put_item(
            Item={
                "prediction_id": prediction_id,
                "prediction": prediction,
                "non_default_probability": Decimal(
                    str(non_default_probability)
                 ),
                "default_probability": Decimal(
                    str(default_probability)
                ),
                "timestamp": timestamp,
            }
        )


        # ----------------------------------------------------
        # 10. Prepare response
        # ----------------------------------------------------

        response = {
            "prediction_id": prediction_id,
            "prediction": prediction,
            "non_default_probability": non_default_probability,
            "default_probability": default_probability,
            "timestamp": timestamp,
        }


        # ----------------------------------------------------
        # 11. Return response
        # ----------------------------------------------------

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(response),
        }


    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(
                {
                    "error": "Prediction failed",
                    "message": str(error),
                }
            ),
        }