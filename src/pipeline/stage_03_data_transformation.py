from src.config.configuration import ConfigurationManager

from src.components.data_transformation import (
    DataTransformation,
)

from src.logger.logger import logger


STAGE_NAME = "Data Transformation"


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> Stage started: {STAGE_NAME} <<<<<<"
        )

        # -----------------------------------------
        # STEP 1: Load transformation configuration
        # -----------------------------------------

        config_manager = ConfigurationManager()

        data_transformation_config = (
            config_manager.get_data_transformation_config()
        )

        # -----------------------------------------
        # STEP 2: Create transformation component
        # -----------------------------------------

        data_transformation = DataTransformation(
            config=data_transformation_config
        )

        # -----------------------------------------
        # STEP 3: Split the data
        # -----------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = data_transformation.split_data()

        # -----------------------------------------
        # STEP 4: Transform and save the data
        # -----------------------------------------

        transformation_artifact = (
            data_transformation.transform_data(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
        )

        # -----------------------------------------
        # STEP 5: Log output artifact paths
        # -----------------------------------------

        logger.info(
            f"Transformed train data: "
            f"{transformation_artifact.transformed_train_path}"
        )

        logger.info(
            f"Transformed test data: "
            f"{transformation_artifact.transformed_test_path}"
        )

        logger.info(
            f"Preprocessor saved at: "
            f"{transformation_artifact.preprocessor_path}"
        )

        logger.info(
            f">>>>>> Stage completed: {STAGE_NAME} <<<<<<"
        )

    except Exception:

        logger.exception(
            f"Stage failed: {STAGE_NAME}"
        )

        raise