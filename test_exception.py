from src.exception.exception import CustomException
from src.logger.logger import logger
import sys

try:
    a = 10 / 0

except Exception as e:
    custom_error = CustomException(e, sys)

    logger.error(custom_error)

    print(custom_error)