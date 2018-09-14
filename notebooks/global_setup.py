# Path modules
import os
import sys
import platform
# Logging modules
import json
import logging
import logging.config

# Set working direction to project root

#git_root = git.Repo('.', search_parent_directories=True).git.rev_parse("--show-toplevel") # '.' causes issue on windows/osx?
root_directory = os.path.realpath("..")
os.chdir(root_directory)
sys.path.insert(0, root_directory)

# Set up logging

LOG_CONFIG_PATH = "notebooks/logger_config.json"

def setup_logging(
    default_level=logging.INFO):
    """
        Load logging configuration
    """
    global LOG_CONFIG_PATH

    if os.path.exists(LOG_CONFIG_PATH):
        with open(LOG_CONFIG_PATH, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def handle_exception(exc_type, exc_value, exc_traceback):
    """
        Function used as an exception handler
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception ({} {} {})".format(platform.system(), platform.release(), os.name), exc_info=(exc_type, exc_value, exc_traceback))

# Run setup
setup_logging()
# Create logger object
logger = logging.getLogger("main_logger")
# Attach the except hook
sys.excepthook = handle_exception

