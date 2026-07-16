import os
import sys
import logging
from datetime import datetime

# Define log file name and directory
LOG_FILE_NAME: str = f"{datetime.now().strftime('%Y_%m-%d_%H-%M-%S')}.log"
LOG_DIR_NAME: str = "logs"

# Construct absolute path for log file
LOG_DIR_PATH: str = os.path.join(os.getcwd(), LOG_DIR_NAME)
os.makedirs(LOG_DIR_PATH, exist_ok=True)

LOG_FILE_PATH: str = os.path.join(LOG_DIR_PATH, LOG_FILE_NAME)

# Logging configuration
logging.basicConfig(
    format="[%(asctime)s] %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# Export the logger object for universal use
logger: logging.Logger = logging.getLogger("EnterpriseCreditRiskLogger")
