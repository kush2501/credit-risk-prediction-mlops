import dagshub
import mlflow


# Step 1: Connect DagsHub with MLflow
dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)


# Step 2: Define champion model URI
MODEL_URI = "models:/CreditRiskModel@champion"


# Step 3: Load champion model
model = mlflow.pyfunc.load_model(MODEL_URI)


# Step 4: Verify
print("====================================")
print("CHAMPION MODEL LOADED SUCCESSFULLY")
print("====================================")
print("Model URI:", MODEL_URI)
print("Model:", model)