import sys

from src.config.configuration import ConfigurationManager

from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import ModelExperimentArtifact

from src.exception.exception import CustomException

from src.logger.logger import logger


if __name__ == "__main__":

    try:

        logger.info(
            ">>>>>> Stage started: Model Evaluation <<<<<<"
        )

        # Configuration Manager
        config_manager = ConfigurationManager()

        # Get Model Evaluation Config
        model_evaluation_config = (
            config_manager.get_model_evaluation_config()
        )

        # Recreate Model Experiment Artifact
        model_experiment_artifact = (
            ModelExperimentArtifact(
                best_model_name="XGBoost",
                best_model_f1_score=0.8334,
                best_model_path=(
                    "artifacts/model_experimentation/"
                    "best_model.pkl"
                ),
            )
        )

        # Initialize Model Evaluation
        model_evaluation = ModelEvaluation(
            config=model_evaluation_config,
            model_experiment_artifact=(
                model_experiment_artifact
            ),
        )

        # Run Evaluation
        model_evaluation_artifact = (
            model_evaluation.evaluate_model()
        )

        logger.info(
            f"Model accepted: "
            f"{model_evaluation_artifact.is_model_accepted}"
        )

        logger.info(
            f"Model F1-score: "
            f"{model_evaluation_artifact.f1_score}"
        )

        logger.info(
            f"Minimum required F1-score: "
            f"{model_evaluation_artifact.minimum_f1_score}"
        )

        logger.info(
            ">>>>>> Stage completed: Model Evaluation <<<<<<"
        )

    except Exception as e:

        logger.exception(
            "Stage failed: Model Evaluation"
        )

        raise CustomException(e, sys)