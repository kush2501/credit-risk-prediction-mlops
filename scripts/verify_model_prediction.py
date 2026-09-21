import pandas as pd
import dagshub
import mlflow

from src.utils.common import load_object


# =========================================================
# Step 1: Connect DagsHub with MLflow
# =========================================================

dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)


# =========================================================
# Step 2: Load saved preprocessor
# =========================================================

preprocessor_path = (
    "artifacts/data_transformation/preprocessor.pkl"
)

preprocessor = load_object(preprocessor_path)

print("Preprocessor loaded successfully")


# =========================================================
# Step 3: Load champion model
# =========================================================

MODEL_URI = "models:/CreditRiskModel@champion"

model = mlflow.pyfunc.load_model(MODEL_URI)

print("Champion model loaded successfully")


# =========================================================
# Step 4: Create new customer data
# =========================================================

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


print("\nCustomer input:")
print(customer_data)


# =========================================================
# Step 5: Transform customer data
# =========================================================

transformed_data = preprocessor.transform(
    customer_data
)

print("\nTransformed shape:", transformed_data.shape)


# =========================================================
# Step 6: Make prediction
# =========================================================

prediction = model.predict(transformed_data)


# =========================================================
# Step 7: Display result
# =========================================================

print("\n====================================")
print("PREDICTION SUCCESSFUL")
print("====================================")

print("Prediction:", prediction)