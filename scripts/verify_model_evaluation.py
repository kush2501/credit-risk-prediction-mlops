from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import ModelTrainerArtifact


if __name__ == "__main__":

    # Get model evaluation configuration
    config_manager = ConfigurationManager()

    model_evaluation_config = (
        config_manager.get_model_evaluation_config()
    )

    # Create sample model training artifact
    model_trainer_artifact = ModelTrainerArtifact(
        trained_model_path="artifacts/model_training/model.pkl",
        accuracy=0.9362,
        precision=0.9809,
        recall=0.7215,
        f1_score=0.8314,
    )

    # Create ModelEvaluation object
    model_evaluation = ModelEvaluation(
        config=model_evaluation_config,
        model_trainer_artifact=model_trainer_artifact,
    )

    # Evaluate model
    model_evaluation_artifact = (
        model_evaluation.evaluate_model()
    )

    print("\nModel Evaluation Artifact:")
    print(model_evaluation_artifact)