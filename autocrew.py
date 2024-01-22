# Filename: autocrew.py

import os
import sys
from pathlib import Path
import configparser

# Add the scripts directory to the system path
scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.append(scripts_path)

from autocrew_core import main

# Read the config file
config = configparser.ConfigParser()
config.read(os.path.join(Path.home(), "autocrew", "config.ini"))

# Pass the config object to the main function
if __name__ == '__main__':
    main(config)
