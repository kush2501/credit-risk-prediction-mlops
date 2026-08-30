from src.config.configuration import ConfigurationManager
from src.components.model_registry import ModelRegistry
from src.logger.logger import logger


STAGE_NAME = "Model Registry"


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> Stage started: {STAGE_NAME} <<<<<<"
        )

        # Step 1: Create configuration manager
        config_manager = ConfigurationManager()

        # Step 2: Get model registry configuration
        model_registry_config = (
            config_manager.get_model_registry_config()
        )

        # Step 3: Create ModelRegistry object
        model_registry = ModelRegistry(
            config=model_registry_config
        )

        # Step 4:
        # Temporary placeholder for the MLflow model URI
        model_uri = model_registry.get_model_uri()

        # Step 5: Check whether the model passed evaluation
        is_model_accepted = (
            model_registry.check_model_acceptance()
        )

        if not is_model_accepted:
            logger.info(
                "Model rejected. Registration skipped."
            )
            raise RuntimeError(
                "Model did not pass evaluation."
            )

        logger.info(
            "Model accepted. Proceeding with registration."
        )

        # Step 6: Register model
        registered_model = (
            model_registry.register_model(
                model_uri=model_uri
            )
        )

        # Step 7: Set champion alias
        model_registry.set_champion_alias(
            registered_model
        )

        # Step 8: Log registration result
        logger.info(
            f"Registered model: "
            f"{model_registry.config.model_name}"
        )

        logger.info(
            f"Registered model version: "
            f"{registered_model}"
        )

        logger.info(
            f">>>>>> Stage completed: "
            f"{STAGE_NAME} <<<<<<"
        )

    except Exception:

        logger.exception(
            f"Stage failed: {STAGE_NAME}"
        )

        raise