import configparser
import subprocess
import pkg_resources
import sys

def check_and_install_dependencies():
    requirements_path = "/home/ai/autocrew/requirements.txt"  # Path to the requirements.txt file
    with open(requirements_path, "r") as f:
        required_packages = [line.strip() for line in f.readlines()]

    for package in required_packages:
        if not package or package.startswith('#'):
            continue  # Skip empty lines or comments
        try:
            pkg_resources.require(package)
        except pkg_resources.DistributionNotFound:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except pkg_resources.VersionConflict:
            pass  # Handle version conflicts or consider upgrading the package

check_and_install_dependencies()

import os
import sys
from pathlib import Path

import argparse
import logging
import traceback

# Add the scripts directory to the system path
scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.append(scripts_path)

# Set up logging to a file
logging.basicConfig(filename='autocrew.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import agent_data
import script_generation

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
        # Rest of your script logic...
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
