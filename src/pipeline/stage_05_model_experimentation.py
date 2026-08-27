from src.config.configuration import ConfigurationManager

from src.components.model_experimentation import (
    ModelExperimentation,
)

from src.logger.logger import logger


STAGE_NAME = "Model Experimentation"


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> Stage started: {STAGE_NAME} <<<<<<"
        )

        # Step 1: Create configuration manager
        config_manager = ConfigurationManager()

        # Step 2: Get experimentation configuration
        model_experiment_config = (
            config_manager.get_model_experiment_config()
        )

        # Step 3: Create experimentation object
        model_experimentation = (
            ModelExperimentation(
                config=model_experiment_config
            )
        )

        # Step 4: Run complete experimentation workflow
        model_experiment_artifact = (
            model_experimentation.initiate_model_experimentation()
        )

        # Step 5: Log final results
        logger.info(
            f"Best model: "
            f"{model_experiment_artifact.best_model_name}"
        )

        logger.info(
            f"Best model F1-score: "
            f"{model_experiment_artifact.best_model_f1_score:.4f}"
        )

        logger.info(
            f"Best model saved at: "
            f"{model_experiment_artifact.best_model_path}"
        )

        logger.info(
            f">>>>>> Stage completed: {STAGE_NAME} <<<<<<"
        )

    except Exception:

        logger.exception(
            f"Stage failed: {STAGE_NAME}"
        )

        raise