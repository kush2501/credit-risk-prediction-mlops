import pandas as pd
import dagshub
import mlflow

from src.utils.common import load_object


def test_model_prediction():

    # Connect DagsHub with MLflow
    dagshub.init(
        repo_owner="kush2501",
        repo_name="credit-risk-prediction-mlops",
        mlflow=True,
    )

    # Load saved preprocessor
    preprocessor_path = (
        "artifacts/data_transformation/preprocessor.pkl"
    )

    preprocessor = load_object(preprocessor_path)

    # Load champion model
    model_uri = "models:/CreditRiskModel@champion"

    model = mlflow.pyfunc.load_model(model_uri)

    # Create customer data
    customer_data = pd.DataFrame(
        [
            {
                "person_age": 35,
                "person_income": 50000,
                "person_home_ownership": "RENT",
                "person_emp_length": 5,
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "loan_amnt": 10000,
                "loan_int_rate": 10.5,
                "loan_percent_income": 0.20,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 8,
            }
        ]
    )

    # Verify raw input
    assert customer_data.shape == (1, 11)

    # Transform customer data
    transformed_data = preprocessor.transform(
        customer_data
    )

    # Verify transformed data
    assert transformed_data.shape == (1, 19)

    # Make prediction
    prediction = model.predict(transformed_data)

    # Verify prediction output
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]