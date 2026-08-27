from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.logger.logger import logger


STAGE_NAME = "Data Ingestion"


def main():
    logger.info(f">>>>>> Stage started: {STAGE_NAME} <<<<<<")

    config_manager = ConfigurationManager()

    data_ingestion_config = (
        config_manager.get_data_ingestion_config()
    )

    data_ingestion = DataIngestion(
        config=data_ingestion_config
    )

    data_ingestion_artifact = (
        data_ingestion.initiate_data_ingestion()
    )

    logger.info(
        f"Raw data saved at: "
        f"{data_ingestion_artifact.raw_data_path}"
    )

    logger.info(
        f">>>>>> Stage completed: {STAGE_NAME} <<<<<<"
    )


if __name__ == "__main__":
    main()