from pathlib import Path
import yaml

from src.exception.exception import CustomException
import sys


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