from src.config.configuration import ConfigurationManager

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

from src.logger.logger import logger


STAGE_NAME = "Data Validation"


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> Stage started: {STAGE_NAME} <<<<<<"
        )

        # Configuration Manager
        config_manager = ConfigurationManager()


        # =========================
        # Get Data Ingestion Config
        # =========================

        data_ingestion_config = (
            config_manager.get_data_ingestion_config()
        )


        # =========================
        # Create Data Ingestion Object
        # =========================

        data_ingestion = DataIngestion(
            config=data_ingestion_config
        )


        # =========================
        # Run Data Ingestion
        # =========================

        ingestion_artifact = (
            data_ingestion.initiate_data_ingestion()
        )


        # =========================
        # Get Data Validation Config
        # =========================

        data_validation_config = (
            config_manager.get_data_validation_config()
        )


        # =========================
        # Create Data Validation Object
        # =========================

        data_validation = DataValidation(
            config=data_validation_config,
            ingestion_artifact=ingestion_artifact
        )


        # =========================
        # Run Data Validation
        # =========================

        validation_artifact = (
            data_validation.validate_dataset()
        )


        logger.info(
            f"Validation status: "
            f"{validation_artifact.validation_status}"
        )


        logger.info(
            f">>>>>> Stage completed: {STAGE_NAME} <<<<<<"
        )


    except Exception:

        logger.exception(
            f"Stage failed: {STAGE_NAME}"
        )

        raise