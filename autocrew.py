# Filename: autocrew.py

import os
import sys
from pathlib import Path
import configparser

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

# Pass the config object to the main function
if __name__ == '__main__':
    try:
        crew_tasks, overall_goal, csv_file_paths, args = main_agent_data(config)
        main_script_generation(crew_tasks, overall_goal, csv_file_paths, args)
    except Exception as e:
        print(f"Error running autocrew.py: {e}")
