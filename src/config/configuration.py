from pathlib import Path

from src.utils.common import read_yaml
from src.constants import (
    CONFIG_FILE_PATH,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig
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