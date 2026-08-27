from src.config.configuration import ConfigurationManager

from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    DataTransformationArtifact,
)

from src.logger.logger import logger


STAGE_NAME = "Model Trainer"


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> Stage started: {STAGE_NAME} <<<<<<"
        )

        # Create configuration manager
        config_manager = ConfigurationManager()

        # Get model trainer configuration
        model_trainer_config = (
            config_manager.get_model_trainer_config()
        )

        # Get data transformation configuration
        data_transformation_config = (
            config_manager.get_data_transformation_config()
        )

        # Create transformation artifact
        data_transformation_artifact = (
            DataTransformationArtifact(

                transformed_train_path=(
                    data_transformation_config.train_data_path
                ),

                transformed_test_path=(
                    data_transformation_config.test_data_path
                ),

                preprocessor_path=(
                    data_transformation_config.preprocessor_path
                ),

            )
        )

        # Initialize model trainer
        model_trainer = ModelTrainer(

            config=model_trainer_config,

            data_transformation_artifact=(
                data_transformation_artifact
            ),

        )

        # Run model training pipeline
        model_trainer_artifact = (
            model_trainer.initiate_model_trainer()
        )

        # Display final metrics
        logger.info(
            f"Model Accuracy: "
            f"{model_trainer_artifact.accuracy:.4f}"
        )

        logger.info(
            f"Model Precision: "
            f"{model_trainer_artifact.precision:.4f}"
        )

        logger.info(
            f"Model Recall: "
            f"{model_trainer_artifact.recall:.4f}"
        )

        logger.info(
            f"Model F1 Score: "
            f"{model_trainer_artifact.f1_score:.4f}"
        )

        logger.info(
            f"Trained model path: "
            f"{model_trainer_artifact.trained_model_path}"
        )

        logger.info(
            f">>>>>> Stage completed: {STAGE_NAME} <<<<<<"
        )

    except Exception:

        logger.exception(
            f"Stage failed: {STAGE_NAME}"
        )

        raise