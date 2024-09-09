import logging
import logging.handlers
from datetime import datetime
import os

# Define the path to the Logs folder
LOGS_FOLDER = os.path.join(os.path.dirname(__file__), 'Logs')

# Create the Logs folder if it does not exist
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Generate a dynamic log filename based on the current date, time, and script name
def getLogfile():
    script_name = os.path.splitext(os.path.basename(__file__))[0]  # Get the script name without the extension
    return os.path.join(LOGS_FOLDER, f"{script_name}_logfile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def setup_logger(filename):
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=5*1024*1024, backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s:%(lineno)d - [Thread-%(thread)d] - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Apply logging configuration
logger = setup_logger(getLogfile())
