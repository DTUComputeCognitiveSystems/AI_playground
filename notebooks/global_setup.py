# Path modules
import os
import sys
from pathlib import Path
# Logging modules
import platform
import json
import logging
import logging.config
# Ipython
from IPython import get_ipython

# Configurations
ROOT_DIRECTORY_NAME = "AI_playground"
LOG_FILE_PATH = "errors.log"
LOG_CONFIG_PATH = "notebooks/logger_config.json"

# Defining functions used in setup

def findRootDir(path, dirname):
    """
        Set working directory to project root
    """
    path = Path(path)
    if os.path.basename(path.parent) == dirname:
        return os.path.realpath(path.parent)
    else:
        return findRootDir(path.parent, dirname)

def run_from_ipython():
    """
        Detect whether we run the file from iPython or not
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def handle_exception(exc_type, exc_value, exc_traceback):
    """
        Function used as an exception handler
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception ({} {} {})".format(platform.system(), platform.release(), os.name), exc_info=(exc_type, exc_value, exc_traceback))

# Changing the root directory
root_directory = findRootDir(os.path.realpath("__file__"), ROOT_DIRECTORY_NAME)
os.chdir(root_directory)
sys.path.insert(0, root_directory)

# Set up logging
# If running the script from iPython console, then we redirect the stderr stream to the file. 
# Else set up logging manually
if run_from_ipython():
    # Stream errors into file
    my_stderr = sys.stderr = open(LOG_FILE_PATH, 'a')
    ipython = get_ipython()
    # Remove colors, otherwise log is unreadable
    ipython.magic("%colors NoColor")
    logger_formatter = logging.Formatter('%(asctime)s - ({} {} {}) %(name)s - %(levelname)s - %(message)s'.format(platform.system(), platform.release(), os.name))
    ipython.log.handlers[0].stream = my_stderr
    ipython.log.handlers[0].setFormatter(logger_formatter)
    ipython.log.setLevel(logging.INFO)
else:
    # Run setup
    config = json.load(open(LOG_CONFIG_PATH, 'rt'))
    logging.config.dictConfig(config)
    # Create logger object
    logger = logging.getLogger("main_logger")
    # Attach the except hook
    sys.excepthook = handle_exception
