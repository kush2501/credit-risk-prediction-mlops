import sys
import json
import mlflow

from src.entity.config_entity import ModelRegistryConfig
from src.exception.exception import CustomException
from src.logger.logger import logger


class ModelRegistry:

    """
    Handles registration of the final selected model
    in the MLflow Model Registry.
    """

    def __init__(
        self,
        config: ModelRegistryConfig,
    ):
        """
        Initialize ModelRegistry with configuration.
        """

        self.config = config

    def register_model(
    self,
    model_uri: str,
    ):
        """
        Register the logged MLflow model into the MLflow Model Registry.

        If the same MLflow run is already registered, reuse the
        existing model version instead of creating a duplicate version.
        """
        try:
            logger.info("Model Registration Started")

            logger.info(
                f"Model URI: {model_uri}"
            )

            logger.info(
                f"Registered model name: "
                f"{self.config.model_name}"
            )

            # Extract run ID from:
            # runs:/<run_id>/final_model
            run_id = model_uri.split("/")[1]

            logger.info(
                f"MLflow run ID: {run_id}"
            )

            # Check existing registered versions
            client = mlflow.MlflowClient()

            existing_versions = client.search_model_versions(
                f"name='{self.config.model_name}'"
            )

            # Check whether this run is already registered
            for version in existing_versions:

                if version.run_id == run_id:

                    logger.info(
                        f"Model already registered."
                    )

                    logger.info(
                        f"Using existing model version: "
                        f"{version.version}"
                    )

                    return int(version.version)

            # Register only if model is not already registered
            logger.info(
                "Model not registered yet. "
                "Creating new model version."
            )

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=self.config.model_name,
            )

            logger.info(
                f"Model registered successfully: "
                f"{registered_model.name}"
            )

            logger.info(
                f"Model version: "
                f"{registered_model.version}"
            )

            logger.info(
                "Model Registration Completed"
            )

            return int(registered_model.version)

        except Exception as e:
            logger.exception(
                "Model Registration Failed"
            )
            raise CustomException(e, sys)
        

    def get_model_uri(self):
        """
        Read the MLflow model URI saved by
        the model experimentation stage.
        """

        try:

            model_uri_path = self.config.mlflow_model_uri_path

            logger.info(
                f"Reading MLflow model URI from: "
                f"{model_uri_path}"
            )

            with open(model_uri_path, "r") as file:
                model_uri = file.read().strip()

            logger.info(
                f"MLflow model URI loaded: {model_uri}"
            )

            return model_uri

        except Exception as e:

            logger.exception(
                "Failed to read MLflow model URI"
            )

            raise CustomException(e, sys)

    def set_champion_alias(self, model_version):
        """
        Assign the champion alias to the registered model version.
        """
        try:
            logger.info(
                f"Setting 'champion' alias for model version: "
                f"{model_version}"
            )

            client = mlflow.MlflowClient()

            client.set_registered_model_alias(
                name=self.config.model_name,
                alias="champion",
                version=model_version,
            )

            logger.info(
                f"'champion' alias assigned successfully to "
                f"{self.config.model_name} version {model_version}"
            )

        except Exception as e:
            logger.exception(
                "Failed to set champion alias"
            )
            raise CustomException(e, sys)


    def check_model_acceptance(self):
        """
        Check whether the model passed evaluation.
        """
        try:
            evaluation_metrics_path = (
                self.config.evaluation_metrics_path
            )

            logger.info(
                f"Reading evaluation metrics from: "
                f"{evaluation_metrics_path}"
            )

            with open(
                evaluation_metrics_path,
                "r"
            ) as file:
                metrics = json.load(file)

            is_model_accepted = metrics[
                "is_model_accepted"
            ]

            logger.info(
                f"Model accepted: {is_model_accepted}"
            )

            return is_model_accepted

        except Exception as e:
            logger.exception(
                "Failed to check model acceptance"
            )
            raise CustomException(e, sys)