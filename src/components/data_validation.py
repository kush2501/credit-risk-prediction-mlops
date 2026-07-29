import os
import sys
import pandas as pd

from src.logger.logger import logger
from src.exception.exception import CustomException

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataIngestionArtifact,
)

from src.utils.common import read_yaml
from src.constants import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(
        self,
        config: DataValidationConfig,
        ingestion_artifact: DataIngestionArtifact
    ):

        self.config = config
        self.ingestion_artifact = ingestion_artifact
        

    def validate_dataset(self):

        try:

            logger.info("Data Validation Started")

            df = pd.read_csv(
                self.ingestion_artifact.raw_data_path
            )

            schema = read_yaml(SCHEMA_FILE_PATH)

            expected_columns = set(schema["columns"].keys())

            dataset_columns = set(df.columns)

            missing_columns = expected_columns - dataset_columns
            extra_columns = dataset_columns - expected_columns

            validation_status = (
                len(missing_columns) == 0 and
                len(extra_columns) == 0
            )

            logger.info(f"Validation Status: {validation_status}")
            logger.info(f"Missing Columns: {list(missing_columns)}")
            logger.info(f"Extra Columns: {list(extra_columns)}")

            # Create validation artifact directory
            os.makedirs(
                self.config.root_dir,
                exist_ok=True
            )

            # Save validation report
            with open(
                self.config.STATUS_FILE,
                "w"
            ) as f:

                f.write(f"Validation Status : {validation_status}\n\n")

                f.write("Missing Columns:\n")
                f.write("-----------------\n")

                if missing_columns:
                    for col in sorted(missing_columns):
                        f.write(f"{col}\n")
                else:
                    f.write("None\n")

                f.write("\nExtra Columns:\n")
                f.write("-----------------\n")

                if extra_columns:
                    for col in sorted(extra_columns):
                        f.write(f"{col}\n")
                else:
                    f.write("None\n")

            # Logging
            logger.info(f"Validation Status : {validation_status}")
            logger.info(f"Missing Columns : {list(missing_columns)}")
            logger.info(f"Extra Columns : {list(extra_columns)}")
            logger.info("Data Validation Completed Successfully")

            return DataValidationArtifact(
                validation_status=validation_status
            )

        except Exception as e:

            logger.exception("Data Validation Failed")

            raise CustomException(e, sys)