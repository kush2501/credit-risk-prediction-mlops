from pathlib import Path
import yaml
import pickle
import sys

from src.exception.exception import CustomException


def create_directories(path_to_directories):
    """
    Create directories if they do not already exist.
    """
    try:
        for path in path_to_directories:
            Path(path).mkdir(
                parents=True,
                exist_ok=True,
            )

    except Exception as e:
        raise CustomException(e, sys)


def read_yaml(path_to_yaml: Path) -> dict:
    """
    Reads a YAML file and returns its contents.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        return content

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj):
    """
    Serialize a Python object and save it to the given file path.
    """
    try:
        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load and deserialize a Python object from the given file path.
    """
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys)