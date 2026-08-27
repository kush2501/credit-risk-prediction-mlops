import sys
import json

from src.entity.config_entity import (
    ModelEvaluationConfig,
    ModelExperimentArtifact,
    ModelEvaluationArtifact,
)

from src.exception.exception import CustomException

from src.logger.logger import logger


class ModelEvaluation:
    """
    Evaluates the final selected model based on
    predefined acceptance criteria.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        model_experiment_artifact: ModelExperimentArtifact,
    ):
        """
        Initialize ModelEvaluation with configuration
        and model experimentation results.
        """

        self.config = config

        self.model_experiment_artifact = (
            model_experiment_artifact
        )


    def evaluate_model(self) -> ModelEvaluationArtifact:
        """
        Evaluate the final selected model, compare its
        F1-score with the minimum required F1-score,
        and save evaluation metrics.
        """

        try:

            logger.info(
                "Model Evaluation Started"
            )

            f1_score = (
                self.model_experiment_artifact
                .best_model_f1_score
            )

            minimum_f1_score = (
                self.config.minimum_f1_score
            )

            logger.info(
                f"Final model F1-Score: {f1_score}"
            )

            logger.info(
                f"Minimum required F1-Score: "
                f"{minimum_f1_score}"
            )

            is_model_accepted = (
                f1_score >= minimum_f1_score
            )

            logger.info(
                f"Model accepted: "
                f"{is_model_accepted}"
            )

            model_evaluation_artifact = (
                ModelEvaluationArtifact(

                    is_model_accepted=(
                        is_model_accepted
                    ),

                    f1_score=f1_score,

                    minimum_f1_score=(
                        minimum_f1_score
                    ),
                )
            )

            # Create output directory
            self.config.evaluation_metrics_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Prepare metrics dictionary
            evaluation_metrics = {

                "is_model_accepted": (
                    is_model_accepted
                ),

                "f1_score": float(
                    f1_score
                ),

                "minimum_f1_score": float(
                    minimum_f1_score
                ),
            }

            # Save metrics as JSON
            with open(
                self.config.evaluation_metrics_path,
                "w",
            ) as file:

                json.dump(
                    evaluation_metrics,
                    file,
                    indent=4,
                )

            logger.info(
                f"Evaluation metrics saved at: "
                f"{self.config.evaluation_metrics_path}"
            )

            logger.info(
                "Model Evaluation Completed"
            )

            return model_evaluation_artifact

        except Exception as e:

            logger.exception(
                "Model Evaluation failed"
            )

            raise CustomException(e, sys)