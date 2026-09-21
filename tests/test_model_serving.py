import dagshub
import mlflow


def test_champion_model_loading():

    # Connect DagsHub with MLflow
    dagshub.init(
        repo_owner="kush2501",
        repo_name="credit-risk-prediction-mlops",
        mlflow=True,
    )

    # Champion model URI
    model_uri = "models:/CreditRiskModel@champion"

    # Load champion model
    model = mlflow.pyfunc.load_model(model_uri)

    # Verify model was loaded
    assert model is not None