from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion

config = ConfigurationManager()

ingestion_config = config.get_data_ingestion_config()

ingestion = DataIngestion(ingestion_config)

artifact = ingestion.initiate_data_ingestion()

print(artifact)