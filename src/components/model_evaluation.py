import sys

from src.entity.config_entity import (
    ModelEvaluationConfig,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

from src.exception.exception import CustomException
from src.logger.logger import logger


class ModelEvaluation:
    """
    Evaluates the trained model based on predefined criteria.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        ):
        """
        Initialize ModelEvaluation with configuration
        and model training results.
        """
        self.config = config
        self.model_trainer_artifact = model_trainer_artifact


    def evaluate_model(self) -> ModelEvaluationArtifact:
        """
        Evaluate the trained model by comparing its F1-score
        with the minimum required F1-score.
        """
        try:
            logger.info("Model Evaluation Started")

            f1_score = self.model_trainer_artifact.f1_score

            minimum_f1_score = self.config.minimum_f1_score

            logger.info(f"Model F1-Score: {f1_score}")
            logger.info(
                f"Minimum required F1-Score: {minimum_f1_score}"
            )

            is_model_accepted = (
                f1_score >= minimum_f1_score
            )

            logger.info(
                f"Model accepted: {is_model_accepted}"
            )

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                f1_score=f1_score,
                minimum_f1_score=minimum_f1_score,
            )

            logger.info("Model Evaluation Completed")

            return model_evaluation_artifact

        except Exception as e:
            logger.exception("Model Evaluation failed")
            raise CustomException(e)