import sys


from src.exception.exception import CustomException
from pathlib import Path

from src.utils.common import read_yaml, create_directories
from src.constants import (
    CONFIG_FILE_PATH,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig, 
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelExperimentConfig,
)


class ConfigurationManager:

    def __init__(self):

        self.config = read_yaml(Path(CONFIG_FILE_PATH))

    def get_data_ingestion_config(self):

        config = self.config["data_ingestion"]

        return DataIngestionConfig(
            root_dir=config["root_dir"],
            source_file=config["source_file"],
            raw_data_path=config["raw_data_path"]
        )

    def get_data_validation_config(self):

        config = self.config["data_validation"]

        return DataValidationConfig(
            root_dir=config["root_dir"],
            STATUS_FILE=config["STATUS_FILE"]
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config["data_transformation"]

        return DataTransformationConfig(
            root_dir=Path(config["root_dir"]),
            data_path=Path(config["data_path"]),
            train_data_path=Path(config["train_data_path"]),
            test_data_path=Path(config["test_data_path"]),
            preprocessor_path=Path(config["preprocessor_path"]),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config["model_trainer"]


        return ModelTrainerConfig(
            root_dir=Path(config["root_dir"]),  
            trained_model_path=Path(
                config["trained_model_path"]
            ),
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Create and return the configuration required
        for model evaluation.
        """
        try:
            config = self.config["model_evaluation"]

            create_directories([config["root_dir"]])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=Path(config["root_dir"]),
                trained_model_path=Path(config["trained_model_path"]),
                minimum_f1_score=float(config["minimum_f1_score"]),
            )

            return model_evaluation_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_model_experiment_config(self) -> ModelExperimentConfig:
        """
        Create and return the configuration required
        for model experimentation.
        """
        try:
            config = self.config["model_experimentation"]

            create_directories([config["root_dir"]])

            model_experiment_config = ModelExperimentConfig(
                root_dir=Path(config["root_dir"]),
                train_data_path=Path(config["train_data_path"]),
                test_data_path=Path(config["test_data_path"]),
            )

            return model_experiment_config

        except Exception as e:
            raise CustomException(e, sys)