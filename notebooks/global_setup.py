# Path modules
import git
import os
import sys
# Logging modules
import json
import logging
import logging.config

# Set working direction to project root

git_root = git.Repo('.', search_parent_directories=True).git.rev_parse("--show-toplevel") # '.' causes issue on windows/osx?
os.chdir(git_root)
sys.path.insert(0, git_root)

# Set up logging

LOG_CONFIG_PATH = "notebooks/logging.json"

def setup_logging(
    default_level=logging.INFO):
    """
        Setup logging configuration
    """
    global LOG_CONFIG_PATH

    if os.path.exists(LOG_CONFIG_PATH):
        with open(LOG_CONFIG_PATH, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

setup_logging()

