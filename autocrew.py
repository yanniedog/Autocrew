# autocrew.py

import os
import sys
from pathlib import Path
import configparser
import argparse
import logging
import traceback

# Add the scripts directory to the system path
scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.append(scripts_path)

from agent_data import main as main_agent_data
from script_generation import main as main_script_generation

# Read the config file
config = configparser.ConfigParser()
try:
    config.read(os.path.join(Path.home(), "autocrew", "config.ini"))
except Exception as e:
    print(f"Error reading config.ini file: {e}")
    sys.exit(1)

# Create a logger object
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script generation script")
    parser.add_argument("--some_argument", help="Description of the argument", type=str)
    args = parser.parse_args()

    try:
        crew_tasks, overall_goal, csv_file_paths, _ = main_agent_data()  # Remove the 'config' argument here
        main_script_generation(crew_tasks, overall_goal, csv_file_paths, args)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
