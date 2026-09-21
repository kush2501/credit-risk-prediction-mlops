import dagshub
import mlflow

dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)

print("Tracking URI:")
print(mlflow.get_tracking_uri())

print("\n========== LOGGED MODELS ==========")

models = mlflow.search_logged_models(
    experiment_ids=["1"],
    output_format="list",
)

if not models:
    print("No Logged Models found.")
else:
    for model in models:
        print("\nModel ID:", model.model_id)
        print("Model Name:", model.name)
        print("Source Run ID:", model.source_run_id)
        print("Model URI:", model.model_uri)
        print("Artifact Location:", model.artifact_location)
        print("Status:", model.status)
        print("--------------------------------")