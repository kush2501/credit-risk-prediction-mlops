import dagshub
import mlflow

dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)

client = mlflow.MlflowClient()

champion = client.get_model_version_by_alias(
    "CreditRiskModel",
    "champion",
)

print("========== CHAMPION MODEL ==========")
print("Model Name:", champion.name)
print("Version:", champion.version)
print("Alias:", champion.aliases)
print("Source:", champion.source)
print("Status:", champion.status)
print("====================================")