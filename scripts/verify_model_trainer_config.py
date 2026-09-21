from src.config.configuration import ConfigurationManager


if __name__ == "__main__":

    config_manager = ConfigurationManager()

    config = config_manager.get_model_trainer_config()

    print(config)