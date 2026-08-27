from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    source_file: str
    raw_data_path: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: str
    STATUS_FILE: str

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path

@dataclass
class DataTransformationArtifact:
    """
    Stores paths of artifacts generated during
    data transformation.
    """

    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_path: Path

@dataclass
class ModelTrainerConfig:
    """
    Configuration required for model training.
    """

    root_dir: Path
    trained_model_path: Path

@dataclass
class ModelTrainerArtifact:
    """
    Store the final output produced by the model training component.
    """
    trained_model_path: Path
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class ModelEvaluationConfig:
    minimum_f1_score: float
    evaluation_metrics_path: Path


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    f1_score: float
    minimum_f1_score: float


@dataclass(frozen=True)
class ModelExperimentConfig:
    """
    Configuration required for model experimentation.
    """

    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    best_model_path: Path

@dataclass(frozen=True)
class ModelExperimentArtifact:
    """
    Stores the final output of model experimentation.
    """

    best_model_name: str
    best_model_f1_score: float
    best_model_path: Path