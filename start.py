# filename: start.py

import subprocess
import os
from logging_config import setup_logging

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def clear_log_file(log_file='autocrew.log'):
    """Clear the log file."""
    try:
        open(log_file, 'w').close()
    except Exception as e:
        print(f"Error clearing log file '{log_file}': {e}")


def main():
    # Execute autocrew.py and capture its output
    try:
        process = subprocess.run(['python3', 'autocrew.py'], check=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(process.stdout)  # Print standard output from autocrew.py
        print(process.stderr)  # Print standard error from autocrew.py
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()

