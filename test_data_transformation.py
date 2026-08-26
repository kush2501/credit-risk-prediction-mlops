import pandas as pd

from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src.utils.common import load_object


if __name__ == "__main__":
    config_manager = ConfigurationManager()

    config = config_manager.get_data_transformation_config()

    transformation = DataTransformation(config)

    X_train, X_test, y_train, y_test = transformation.split_data()

    print("\nBefore transformation:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    data_transformation_artifact = (
        transformation.transform_data(
            X_train,
            X_test,
            y_train,
            y_test,
            )
    )
    print("\nData Transformation Artifact:")
    print(
        "Train path:",
        data_transformation_artifact.transformed_train_path,
    )
    print(
        "Test path:",
        data_transformation_artifact.transformed_test_path,
    )
    print(
        "Preprocessor path:",
        data_transformation_artifact.preprocessor_path,
    )

    