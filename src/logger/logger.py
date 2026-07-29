import logging
import os
import sys
from datetime import datetime

# ==============================
# Create Logs Directory
# ==============================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================
# Log File Name
# Example: 2026-07-28.log
# ==============================
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# ==============================
# Log Format
# ==============================
LOG_FORMAT = (
    "[%(asctime)s] | %(levelname)-8s | %(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==============================
# Create Logger
# ==============================
logger = logging.getLogger("CreditRiskPrediction")
logger.setLevel(logging.INFO)

# Prevent duplicate logs
logger.propagate = False

# Remove old handlers (important in Jupyter/re-runs)
if logger.hasHandlers():
    logger.handlers.clear()

# ==============================
# File Handler
# ==============================
file_handler = logging.FileHandler(
    LOG_FILE_PATH,
    encoding="utf-8"
)

file_handler.setLevel(logging.INFO)

# ==============================
# Console Handler
# ==============================
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# ==============================
# Formatter
# ==============================
formatter = logging.Formatter(
    fmt=LOG_FORMAT,
    datefmt=DATE_FORMAT
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# ==============================
# Add Handlers
# ==============================
logger.addHandler(file_handler)
logger.addHandler(console_handler)


