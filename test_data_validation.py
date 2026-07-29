from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

config = ConfigurationManager()

# Data Ingestion
ingestion_config = config.get_data_ingestion_config()
ingestion = DataIngestion(ingestion_config)
ingestion_artifact = ingestion.initiate_data_ingestion()

# Data Validation
validation_config = config.get_data_validation_config()
validation = DataValidation(
    validation_config,
    ingestion_artifact
)

validation_artifact = validation.validate_dataset()

print(validation_artifact)