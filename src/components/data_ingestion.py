import os
import shutil
import sys

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from src.logger.logger import logger
from src.exception.exception import CustomException


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        logger.info("Data Ingestion Started")

        try:

            os.makedirs(self.config.root_dir, exist_ok=True)

            shutil.copy(
                self.config.source_file,
                self.config.raw_data_path
            )

            logger.info("Dataset copied successfully.")

            return DataIngestionArtifact(
                raw_data_path=self.config.raw_data_path
            )

        except Exception as e:
            logger.exception("Data Ingestion Failed")
            raise CustomException(e, sys)