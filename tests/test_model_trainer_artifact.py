from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import DataTransformationArtifact, ModelTrainerArtifact

from pathlib import Path


if __name__ == "__main__":

    model_trainer_artifact = ModelTrainerArtifact(
        trained_model_path=Path(
            "artifacts/model_training/model.pkl"
        ),
        accuracy=0.9362,
        precision=0.9809,
        recall=0.7215,
        f1_score=0.8314,
    )

    print(model_trainer_artifact)