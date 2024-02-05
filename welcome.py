<<<<<<< Updated upstream
#!/usr/bin/env python3

# filename: welcome.py
import configparser
import subprocess
import logging
import logging.handlers
import copy
import io
import os
import re
import csv
import sys
import textwrap
import ollama

from autocrew import check_latest_version, generate_startup_message

from logging_config import flush_log_handlers
from logging_config import setup_logging
from core import AutoCrew

GREEK_ALPHABETS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
                   "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]

def truncate_overall_goal(overall_goal, max_length=40):
    """Truncate the overall goal to a maximum length."""
    return overall_goal[:max_length]  # This should return a string

def get_ranked_crews(overall_goal):
    """Retrieve the list of ranked crews based on the truncated overall goal."""
    script_dir = os.path.join(os.getcwd(), "scripts")
    truncated_goal = truncate_overall_goal(overall_goal)

    # Prepare a pattern to match the relevant part of the filenames
    greek_alphabets = "|".join(GREEK_ALPHABETS)  # Join Greek alphabets with a pipe (|) character for regex alternation
    pattern = re.compile(rf"crewai-autocrew-\d{{8}}-\d{{6}}-{truncated_goal.replace(' ', '-')}-({greek_alphabets})\.csv$")

    # Find all files that match the pattern
    matching_files = [f for f in os.listdir(script_dir) if pattern.match(f)]

    # Map each file to its corresponding Greek alphabet crew name
    ranked_crews = {}
    for file_name in matching_files:
        match = pattern.match(file_name)
        if match:
            greek_alphabet = match.group(1)  # Extract the Greek alphabet suffix from the filename
            first_letter = greek_alphabet[0]  # Get the first letter of the Greek alphabet
            ranked_crews[first_letter] = f"{greek_alphabet.capitalize()} Crew"  # Capitalize the first letter and append "Crew"

    return ranked_crews

def log_initial_config(config):
    """Log the initial configuration settings with sensitive information redacted."""
    redacted_config = copy.deepcopy(config)
    for section in redacted_config.sections():
        for key, value in redacted_config.items(section):
            if 'api_key' in key.lower() or 'auth_token' in key.lower():
                redacted_config.set(section, key, '*' * (len(value) - 4) + value[-4:])
    with io.StringIO() as config_string:
        redacted_config.write(config_string)
        config_string.seek(0)
        logging.debug("Initial config.ini settings (redacted):\n" + config_string.read())

def get_input(prompt, default=None, validator=lambda x: True):
    """Get user input with validation and logging."""
    while True:
        user_input = input(prompt)
        if user_input == '' and default is not None:
            user_input = default
        if validator(user_input):
            logging.debug(f"Prompt: {prompt.strip()} | User input: {user_input}")
            return user_input
        logging.debug(f"Invalid input: {user_input}")
        logging.info("Invalid input, please try again.")

def validate_positive_int(value):
    """Validate if the provided value is a positive integer."""
    try:
        return int(value) > 0
    except ValueError:
        return False

def validate_yes_no(value):
    """Validate if the provided value is a yes/no response."""
    return value.lower() in ['yes', 'no', 'y', 'n']

def select_from_list(options, prompt):
    """Allow the user to select an option from a list."""
    for index, option in enumerate(options, start=1):
        logging.info(f"{index}) {option}")
    while True:
        selection = input(prompt)
        if selection.isdigit() and 0 < int(selection) <= len(options):
            selected_option = options[int(selection) - 1]
            logging.debug(f"Prompt: {prompt.strip()} | User input: {selection}")
            logging.debug(f"Selected option: {selected_option}")
            return selected_option
        logging.debug(f"Invalid input: {selection}")
        logging.info("Invalid selection, please try again.")

def run_autocrew_script(num_alternative_crews, overall_goal, rank_crews):
    # Set environment variables for autocrew.py
    os.environ['LOG_LEVEL'] = 'CONFIG'
    os.environ['LOG_FILE'] = 'autocrew.log'
    os.environ['CALLED_FROM_WELCOME'] = '1'

    args = ['python3', 'autocrew.py', '-m', str(num_alternative_crews), '-r', overall_goal] if rank_crews else ['python3', 'autocrew.py', '-m', str(num_alternative_crews), overall_goal]
    try:
        # Start the subprocess and get the output in real-time
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
            print_ranking_csv_needed = False
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end='')
                    logging.debug(line.strip())  # Log to file
                    if "See here for details:" in line:
                        print_ranking_csv_needed = True

        # Check the return code to determine if the subprocess was successful
        if process.returncode != 0:
            logging.error("Autocrew script returned a non-zero exit code.")
            return False

        ##### if print_ranking_csv_needed and rank_crews:
        #####    print_ranking_csv(overall_goal)

        return True

    except Exception as e:
        # Log any exceptions that occur while running the subprocess
        logging.exception("Exception in running autocrew.py: %s", str(e))
        return False

    finally:
        # Clear the environment variable after running
        del os.environ['CALLED_FROM_WELCOME']

def get_user_selected_crew(ranked_crews):
    """Get the crew selected by the user."""
    logging.info("Select the crew you wish to run:")
    sorted_crews = sorted(ranked_crews.items())
    for letter, crew_name in sorted_crews:
        logging.info(f"{letter}) {crew_name}")

    while True:
        selected_letter = input("Enter your choice (letter): ").lower()
        if selected_letter in ranked_crews:
            return selected_letter
        else:
            logging.info("Invalid selection, please try again using the letter of the Greek alphabet.")

def find_script_path(truncated_goal, selected_letter, script_dir):
    """Find the script path based on the truncated goal and selected letter."""
    script_pattern = re.compile(rf"crewai-autocrew-\d{{8}}-\d{{6}}-{truncated_goal}-({selected_letter})\.py$")
    for file_name in os.listdir(script_dir):
        if script_pattern.match(file_name):
            return os.path.join(script_dir, file_name)
    return None

def execute_script(script_path):
    """Execute the selected script."""
    try:
        logging.debug(f"Executing script: {script_path}")
        subprocess.run(['python3', script_path], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while executing the script: {e}")
    except FileNotFoundError:
        logging.error(f"The script file was not found: {script_path}")

def handle_ranked_crews(overall_goal):
    ranked_crews = get_ranked_crews(overall_goal)
    if not ranked_crews:
        logging.info("No ranked crews available for selection.")
        flush_log_handlers()
        return

    selected_letter = get_user_selected_crew(ranked_crews)

    truncated_goal = truncate_overall_goal(overall_goal)
    greek_name = {'a': 'alpha', 'b': 'beta', 'g': 'gamma'}[selected_letter]

    script_dir = os.path.join(os.getcwd(), "scripts")
    script_path = find_script_path(truncated_goal, greek_name, script_dir)

    if not script_path:
        logging.info(f"No script file found for the selected crew: {ranked_crews[selected_letter]}")
        flush_log_handlers()
        return

    execute_script(script_path)
    flush_log_handlers()

def save_configuration(config):
    """Save the current configuration to the config.ini file."""
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    logging.debug("Configuration saved to config.ini.")

def get_redacted_api_key(api_key):
    """Redact all but the last 4 characters of the API key."""
    return '*' * (len(api_key) - 4) + api_key[-4:]

def choose_openai_model(config):
    """Allow the user to choose or enter an OpenAI model manually."""
    openai_models = {
        'a': 'gpt-3.5-turbo-1106',
        'b': 'gpt-3.5-turbo',
        'c': 'gpt-4-turbo-preview',
        'd': 'gpt-4-1106-preview',
        'e': 'Enter model manually'
    }
    choice = select_from_list(list(openai_models.values()), "Choose an OpenAI model: ")
    if choice.lower() == 'enter model manually':
        return get_input("Enter the OpenAI model: ")
    else:
        return choice

def handle_openai_api_key(config):
    """Handle the OpenAI API key based on user input."""
    if 'openai_api_key' in config['AUTHENTICATORS'] and config['AUTHENTICATORS']['openai_api_key']:
        redacted_key = get_redacted_api_key(config['AUTHENTICATORS']['openai_api_key'])
        use_existing_key = get_input(f"Use existing OpenAI API key ({redacted_key})? (y/n): ", validator=validate_yes_no)
        if use_existing_key.lower() in ['no', 'n']:
            new_key = get_input("Enter your new OpenAI API key: ")
            config['AUTHENTICATORS']['openai_api_key'] = new_key
    else:
        new_key = get_input("Enter your OpenAI API key: ")
        config['AUTHENTICATORS']['openai_api_key'] = new_key

def choose_llm_endpoint_and_model(config):
    """Prompt the user to choose the LLM endpoint and model, or keep the existing ones."""
    existing_endpoint = config.get('BASIC', 'llm_endpoint', fallback=None)
    existing_model = config.get('OLLAMA_CONFIG' if existing_endpoint == 'ollama' else 'OPENAI_CONFIG', 'openai_model' if existing_endpoint == 'openai' else 'llm_model', fallback=None)

    logging.debug(f"Existing LLM endpoint: {existing_endpoint}")
    logging.debug(f"Existing LLM model: {existing_model}")

    # Default choice is to use existing settings
    use_existing = get_input(f"Use existing settings (LLM endpoint: {existing_endpoint}, Model: {existing_model})? (y/n) [yes]: ", default='yes', validator=validate_yes_no)
    logging.debug(f"User chose to {'use' if use_existing.lower() in ['yes', 'y'] else 'not use'} existing settings.")
    if use_existing.lower() in ['yes', 'y']:
        return existing_endpoint, existing_model

    llm_endpoints = ['ollama', 'openai']
    llm_endpoint = select_from_list(llm_endpoints, "Select the LLM endpoint: ")
    config.set('BASIC', 'llm_endpoint', llm_endpoint)  # Update the config object

    if llm_endpoint == 'openai':
        openai_model = choose_openai_model(config)
        config.set('OPENAI_CONFIG', 'openai_model', openai_model)  # Update the config object
        handle_openai_api_key(config)
    elif llm_endpoint == 'ollama':
        # Call the function from ollama.py here
        ollama_model = ollama.main()  # This should return the selected model
        config.set('OLLAMA_CONFIG', 'llm_model', ollama_model)  # Update the config object
        openai_model = ollama_model  # For consistency in the return statement

    # Ask if the user wants to use the same settings for CrewAI scripts
    use_same_for_crewai = get_input("Use the same settings for CrewAI scripts? (y/n): ", default='y', validator=validate_yes_no)
    if use_same_for_crewai.lower() in ['yes', 'y']:
        config.set('CREWAI_SCRIPTS', 'llm_endpoint_within_generated_scripts', llm_endpoint)
        config.set('CREWAI_SCRIPTS', 'llm_model_within_generated_scripts', openai_model)
    else:
        # Prompt the user to select the CrewAI endpoint and model from the same list again
        crewai_endpoint = select_from_list(llm_endpoints, "Select the CrewAI endpoint for generated scripts: ")
        config.set('CREWAI_SCRIPTS', 'llm_endpoint_within_generated_scripts', crewai_endpoint)
        if crewai_endpoint == 'openai':
            crewai_model = choose_openai_model(config)
            config.set('CREWAI_SCRIPTS', 'llm_model_within_generated_scripts', crewai_model)
        else:
            crewai_model = get_input("Enter the CrewAI model for Ollama: ")
            config.set('CREWAI_SCRIPTS', 'llm_model_within_generated_scripts', crewai_model)

    return llm_endpoint, openai_model

def clear_screen_and_logfile(logfile):
    """Clear the screen and the log file."""
    # Clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Clear the log file
    with open(logfile, 'w'):
        pass

def print_ranking_csv(overall_goal):
    """Print and log the contents of the ranking CSV file line by line."""
    script_dir = os.path.join(os.getcwd(), "scripts")
    truncated_goal = truncate_overall_goal(overall_goal)
    pattern = re.compile(rf"crewai-autocrew-\d{{8}}-\d{{6}}-{truncated_goal}-ranking\.csv$")

    # Find the ranking CSV file
    for file_name in os.listdir(script_dir):
        if pattern.match(file_name):
            ranking_csv_path = os.path.join(script_dir, file_name)
            break
    else:
        log_message = "Ranking CSV file not found."
        logging.error(log_message)
        print(log_message)
        return

    try:
        with open(ranking_csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                line = ", ".join(row)
                logging.info(line)  # Log the line
                print(line)  # Print the line

    except Exception as e:
        log_message = f"Error reading CSV file: {e}"
        logging.error(log_message)
        print(log_message)



def get_max_widths(headers, data, max_width):
    """Calculate the maximum width for each column."""
    widths = [min(len(str(header)), max_width) for header in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = min(max(widths[i], len(str(cell))), max_width)
    return widths

def print_table(headers, data, widths):
    """Print the table with given headers, data, and column widths."""
    header_row = "|".join(str(header).ljust(width) for header, width in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    for row in data:
        print("|".join(str(cell).ljust(width) for cell, width in zip(row, widths)))


def main():
    setup_logging()
    config = configparser.ConfigParser()
    config.read('config.ini')
    log_initial_config(config)

    # Check for the latest version and generate startup message
    latest_version, version_message = check_latest_version()
    startup_message = generate_startup_message(latest_version, version_message)
    logging.info(startup_message)

    overall_goal = get_input("Please specify your overall goal: ")
    
    # Default answer set to 3 for the number of alternative crews
    num_alternative_crews = get_input("How many alternative crews do you wish to generate? [3]: ", default='3', validator=validate_positive_int)
    
    # Default answer set to 'yes' for ranking
    rank_crews = get_input("Do you want the crews to be ranked afterwards? (yes/no) [yes]: ", default='yes', validator=validate_yes_no) in ['yes', 'y']

    # Choose LLM Endpoint and Model or use existing settings
    llm_endpoint, llm_model = choose_llm_endpoint_and_model(config)

    # Automatically save the settings to config.ini
    save_configuration(config)

    # Create an instance of AutoCrew and log the updated configuration
    config_path = os.path.join(os.getcwd(), 'config.ini')  # Specify the path to the config.ini file
    autocrew = AutoCrew(config_path)  # Pass the path to the config.ini file to the AutoCrew constructor
    autocrew.log_config_with_redacted_api_keys()

    # Execute AutoCrew script and handle rankings if applicable
    if run_autocrew_script(num_alternative_crews, overall_goal, rank_crews):
        if rank_crews:
            handle_ranked_crews(overall_goal)
            print_ranking_csv(overall_goal)  # Print the ranking CSV file after ranking is completed
        else:
            # Additional logic if ranking is not performed
            pass
    else:
        logging.error("Autocrew script execution failed.")

if __name__ == "__main__":
    clear_screen_and_logfile('autocrew.log')
    main()
=======
#!/usr/bin/env python3

# filename: welcome.py
import configparser
import subprocess
import logging
import logging.handlers
import copy
import io
import os
import re
import csv
import sys
import textwrap
import time
import ollama
import ngrok

from autocrew import check_latest_version, generate_startup_message
from ollama import main as ollama_main
from ollama import pull_model
from logging_config import setup_logging, flush_log_handlers
from core import AutoCrew
from ngrok import get_ngrok_api_key, get_ngrok_tunnels, get_public_url, get_colab_notebook_url, display_ngrok_setup_instructions

GREEK_ALPHABETS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
                   "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]

ngrok_tunnel_info = None

def truncate_overall_goal(overall_goal, max_length=40):
    """Truncate the overall goal to a maximum length."""
    return overall_goal[:max_length]  # This should return a string

def get_ranked_crews(overall_goal):
    """Retrieve the list of ranked crews based on the truncated overall goal."""
    script_dir = os.path.join(os.getcwd(), "scripts")
    truncated_goal = truncate_overall_goal(overall_goal)

    # Prepare a pattern to match the relevant part of the filenames
    greek_alphabets = "|".join(GREEK_ALPHABETS)  # Join Greek alphabets with a pipe (|) character for regex alternation
    pattern = re.compile(rf"crewai-autocrew-\d{{8}}-\d{{6}}-{truncated_goal.replace(' ', '-')}-({greek_alphabets})\.csv$")

    # Find all files that match the pattern
    matching_files = [f for f in os.listdir(script_dir) if pattern.match(f)]

    # Map each file to its corresponding Greek alphabet crew name
    ranked_crews = {}
    for file_name in matching_files:
        match = pattern.match(file_name)
        if match:
            greek_alphabet = match.group(1)  # Extract the Greek alphabet suffix from the filename
            first_letter = greek_alphabet[0]  # Get the first letter of the Greek alphabet
            ranked_crews[first_letter] = f"{greek_alphabet.capitalize()} Crew"  # Capitalize the first letter and append "Crew"

    return ranked_crews

def log_initial_config(config):
    """Log the initial configuration settings with sensitive information redacted."""
    redacted_config = copy.deepcopy(config)
    for section in redacted_config.sections():
        for key, value in redacted_config.items(section):
            if 'api_key' in key.lower() or 'auth_token' in key.lower():
                redacted_config.set(section, key, '*' * (len(value) - 4) + value[-4:])
    with io.StringIO() as config_string:
        redacted_config.write(config_string)
        config_string.seek(0)
        logging.debug("Initial config.ini settings (redacted):\n" + config_string.read())

def get_input(prompt, default=None, validator=None):
    """Get user input with validation and logging."""
    while True:
        user_input = input(prompt)
        if user_input == '' and default is not None:
            user_input = default
            logging.debug(f"Prompt: {prompt.strip()} | User input: {user_input} (default)")
        else:
            logging.debug(f"Prompt: {prompt.strip()} | User input: {user_input}")
        
        # If no validator is provided, or if the validator returns True, return the input
        if validator is None or validator(user_input):
            return user_input
        else:
            logging.debug(f"Invalid input: {user_input}")
            logging.info("Invalid input, please try again.")

            


            

def validate_yes_no(value, default=None):
    """Validate if the provided value is a yes/no response or default."""
    if value == '' and default is not None:
        return True
    return value.lower() in ['yes', 'no', 'y', 'n']


def validate_positive_int(value):
    """Validate if the provided value is a positive integer."""
    try:
        return int(value) > 0
    except ValueError:
        return False



def select_from_list(options, prompt):
    """Allow the user to select an option from a list."""
    for index, option in enumerate(options, start=1):
        logging.info(f"{index}) {option}")
    while True:
        selection = input(prompt)
        if selection.isdigit() and 0 < int(selection) <= len(options):
            selected_option = options[int(selection) - 1]
            logging.debug(f"Prompt: {prompt.strip()} | User input: {selection}")
            logging.debug(f"Selected option: {selected_option}")
            return selected_option
        logging.debug(f"Invalid input: {selection}")
        logging.info("Invalid selection, please try again.")

def run_autocrew_script(num_alternative_crews, overall_goal, rank_crews):
    # Set environment variables for autocrew.py
    os.environ['LOG_LEVEL'] = 'CONFIG'
    os.environ['LOG_FILE'] = 'autocrew.log'
    os.environ['CALLED_FROM_WELCOME'] = '1'

    args = ['python3', 'autocrew.py', '-m', str(num_alternative_crews), '-r', overall_goal] if rank_crews else ['python3', 'autocrew.py', '-m', str(num_alternative_crews), overall_goal]
    try:
        # Start the subprocess and get the output in real-time
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
            print_ranking_csv_needed = False
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end='')
                    logging.debug(line.strip())  # Log to file
                    if "See here for details:" in line:
                        print_ranking_csv_needed = True

        # Check the return code to determine if the subprocess was successful
        if process.returncode != 0:
            logging.error("Autocrew script returned a non-zero exit code.")
            return False

        ##### if print_ranking_csv_needed and rank_crews:
        #####    print_ranking_csv(overall_goal)

        return True

    except Exception as e:
        # Log any exceptions that occur while running the subprocess
        logging.exception("Exception in running autocrew.py: %s", str(e))
        return False

    finally:
        # Clear the environment variable after running
        del os.environ['CALLED_FROM_WELCOME']

def get_user_selected_crew(ranked_crews):
    """Get the crew selected by the user."""
    logging.info("Select the crew you wish to run:")
    sorted_crews = sorted(ranked_crews.items())
    for letter, crew_name in sorted_crews:
        logging.info(f"{letter}) {crew_name}")

    while True:
        selected_letter = input("Enter your choice (letter): ").lower()
        if selected_letter in ranked_crews:
            return selected_letter
        else:
            logging.info("Invalid selection, please try again using the letter of the Greek alphabet.")

def find_script_path(truncated_goal, selected_letter, script_dir):
    """Find the script path based on the truncated goal and selected letter."""
    script_pattern = re.compile(rf"crewai-autocrew-\d{{8}}-\d{{6}}-{truncated_goal}-({selected_letter})\.py$")
    for file_name in os.listdir(script_dir):
        if script_pattern.match(file_name):
            return os.path.join(script_dir, file_name)
    return None

def execute_script(script_path):
    """Execute the selected script."""
    try:
        logging.debug(f"Executing script: {script_path}")
        subprocess.run(['python3', script_path], check=True)
    except subprocess.CalledProcessError as e:
        logging.exception(f"An error occurred while executing the script: {e}")
    except FileNotFoundError:
        logging.exception(f"The script file was not found: {script_path}")


def handle_ranked_crews(overall_goal):
    ranked_crews = get_ranked_crews(overall_goal)
    if not ranked_crews:
        logging.info("No ranked crews available for selection.")
        flush_log_handlers()
        return

    selected_letter = get_user_selected_crew(ranked_crews)

    truncated_goal = truncate_overall_goal(overall_goal)
    greek_name = {'a': 'alpha', 'b': 'beta', 'g': 'gamma'}[selected_letter]

    script_dir = os.path.join(os.getcwd(), "scripts")
    script_path = find_script_path(truncated_goal, greek_name, script_dir)

    if not script_path:
        logging.info(f"No script file found for the selected crew: {ranked_crews[selected_letter]}")
        flush_log_handlers()
        return

    execute_script(script_path)
    flush_log_handlers()

def save_configuration(config):
    """Save the current configuration to the config.ini file."""
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    logging.debug("Configuration saved to config.ini.")

def get_redacted_api_key(api_key):
    """Redact all but the last 4 characters of the API key."""
    return '*' * (len(api_key) - 4) + api_key[-4:]

def choose_openai_model(config):
    """Allow the user to choose or enter an OpenAI model manually."""
    openai_models = {
        'a': 'gpt-3.5-turbo-1106',
        'b': 'gpt-3.5-turbo',
        'c': 'gpt-4-turbo-preview',
        'd': 'gpt-4-1106-preview',
        'e': 'Enter model manually'
    }
    choice = select_from_list(list(openai_models.values()), "Choose an OpenAI model: ")
    if choice.lower() == 'enter model manually':
        return get_input("Enter the OpenAI model: ")
    else:
        return choice

def handle_openai_api_key(config):
    """Handle the OpenAI API key based on user input."""
    if 'openai_api_key' in config['AUTHENTICATORS'] and config['AUTHENTICATORS']['openai_api_key']:
        redacted_key = get_redacted_api_key(config['AUTHENTICATORS']['openai_api_key'])
        use_existing_key = get_input(f"Use existing OpenAI API key ({redacted_key})? (y/n): ", validator=validate_yes_no)
        if use_existing_key.lower() in ['no', 'n']:
            new_key = get_input("Enter your new OpenAI API key: ")
            config['AUTHENTICATORS']['openai_api_key'] = new_key
    else:
        new_key = get_input("Enter your OpenAI API key: ")
        config['AUTHENTICATORS']['openai_api_key'] = new_key

def choose_llm_endpoint_and_model(config):
    """Prompt the user to choose the LLM endpoint and model, or keep the existing ones."""
    existing_endpoint = config.get('BASIC', 'llm_endpoint', fallback=None)
    existing_model = config.get('OLLAMA_CONFIG' if existing_endpoint == 'ollama' else 'OPENAI_CONFIG', 'openai_model' if existing_endpoint == 'openai' else 'llm_model', fallback=None)

    # Check if the remote host setting is enabled
    use_remote_ollama_host = config.getboolean('REMOTE_HOST_CONFIG', 'use_remote_ollama_host', fallback=False)
    endpoint_display = f"{existing_endpoint} (remote)" if use_remote_ollama_host and existing_endpoint == 'ollama' else existing_endpoint

    logging.debug(f"Existing LLM endpoint: {existing_endpoint}")
    logging.debug(f"Existing LLM model: {existing_model}")

    # Default choice is to use existing settings
    use_existing = get_input(f"Use existing settings (LLM endpoint: {endpoint_display}, Model: {existing_model})? (y/n) [yes]: ", default='yes', validator=validate_yes_no)
    logging.debug(f"User chose to {'use' if use_existing.lower() in ['yes', 'y'] else 'not use'} existing settings.")
    if use_existing.lower() in ['yes', 'y']:
        return existing_endpoint, existing_model

    llm_endpoints = ['ollama', 'openai']
    llm_endpoint = select_from_list(llm_endpoints, "Select the LLM endpoint: ")
    config.set('BASIC', 'llm_endpoint', llm_endpoint)  # Update the config object

    if llm_endpoint == 'openai':
        openai_model = choose_openai_model(config)
        config.set('OPENAI_CONFIG', 'openai_model', openai_model)  # Update the config object
        handle_openai_api_key(config)
    elif llm_endpoint == 'ollama':
        base_url = config['REMOTE_HOST_CONFIG'].get('ollama_host')  # Retrieve the base URL
        ollama_model = ollama_main()  # This should return the selected model
        pull_result = pull_model(ollama_model, base_url)  # Pass the base URL as an argument
        if pull_result is None or pull_result.get('status') != 'success':
            logging.error(f"Failed to pull model {ollama_model}.")
            sys.exit(1)
        config.set('OLLAMA_CONFIG', 'llm_model', ollama_model)  # Update the config object
        openai_model = ollama_model  # For consistency in the return statement

    # Ask if the user wants to use the same settings for CrewAI scripts
    use_same_for_crewai = get_input("Use the same settings for CrewAI scripts? (y/n) [yes]: ", default='yes', validator=validate_yes_no)
    if use_same_for_crewai.lower() in ['yes', 'y']:
        config.set('CREWAI_SCRIPTS', 'llm_endpoint_within_generated_scripts', llm_endpoint)
        config.set('CREWAI_SCRIPTS', 'llm_model_within_generated_scripts', openai_model)
    else:
        # Prompt the user to select the CrewAI endpoint and model from the same list again
        crewai_endpoint = select_from_list(llm_endpoints, "Select the CrewAI endpoint for generated scripts: ")
        config.set('CREWAI_SCRIPTS', 'llm_endpoint_within_generated_scripts', crewai_endpoint)
        if crewai_endpoint == 'openai':
            crewai_model = choose_openai_model(config)
            config.set('CREWAI_SCRIPTS', 'llm_model_within_generated_scripts', crewai_model)
        else:
            crewai_model = get_input("Enter the CrewAI model for Ollama: ")
            config.set('CREWAI_SCRIPTS', 'llm_model_within_generated_scripts', crewai_model)

    return llm_endpoint, openai_model



def clear_screen_and_logfile(logfile):
    """Clear the screen and the log file."""
    # Clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Clear the log file
    with open(logfile, 'w'):
        pass

def print_ranking_csv(overall_goal):
    """Print and log the contents of the ranking CSV file line by line."""
    script_dir = os.path.join(os.getcwd(), "scripts")
    truncated_goal = truncate_overall_goal(overall_goal)
    pattern = re.compile(rf"crewai-autocrew-\d{{8}}-\d{{6}}-{truncated_goal}-ranking\.csv$")

    # Find the ranking CSV file
    for file_name in os.listdir(script_dir):
        if pattern.match(file_name):
            ranking_csv_path = os.path.join(script_dir, file_name)
            break
    else:
        log_message = "Ranking CSV file not found."
        logging.error(log_message)
        print(log_message)
        return

    try:
        with open(ranking_csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                line = ", ".join(row)
                logging.info(line)  # Log the line
                print(line)  # Print the line

    except Exception as e:
        log_message = f"Error reading CSV file: {e}"
        logging.error(log_message)
        print(log_message)



def get_max_widths(headers, data, max_width):
    """Calculate the maximum width for each column."""
    widths = [min(len(str(header)), max_width) for header in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = min(max(widths[i], len(str(cell))), max_width)
    return widths

def print_table(headers, data, widths):
    """Print the table with given headers, data, and column widths."""
    header_row = "|".join(str(header).ljust(width) for header, width in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    for row in data:
        print("|".join(str(cell).ljust(width) for cell, width in zip(row, widths)))

def handle_advanced_settings(config):
    try:
        advanced_settings = get_input("Would you like to view advanced settings? (yes/no) [no]: ", default='no', validator=validate_yes_no)
        if advanced_settings.lower() in ['yes', 'y']:
            use_remote_ollama = get_input("Would you like to run Ollama on a cloud server using an ngrok tunnel? (yes/no): ", validator=validate_yes_no)
            if use_remote_ollama.lower() in ['yes', 'y']:
                # Check if there is an existing ngrok API key
                existing_ngrok_key = config.get('AUTHENTICATORS', 'ngrok_api_key', fallback='')
                if existing_ngrok_key:
                    use_existing_key = get_input(f"Use existing ngrok API key ({get_redacted_api_key(existing_ngrok_key)})? (y/n): ", validator=validate_yes_no)
                    if use_existing_key.lower() in ['no', 'n']:
                        new_key = get_input("Enter your new ngrok API key: ", validator=lambda x: x.strip() != '')
                        if new_key:
                            config['AUTHENTICATORS']['ngrok_api_key'] = new_key
                else:
                    new_key = get_input("Enter your ngrok API key: ", validator=lambda x: x.strip() != '')
                    if new_key:
                        config['AUTHENTICATORS']['ngrok_api_key'] = new_key

                # Retrieve ngrok tunnel information
                ngrok_api_key = get_ngrok_api_key()
                tunnels = get_ngrok_tunnels(ngrok_api_key)
                public_url = get_public_url(tunnels)

                if public_url:
                    logging.info(f"Ngrok public URL: {public_url}")
                    # Store the public URL in the config.ini file under REMOTE_HOST_CONFIG section
                    config['REMOTE_HOST_CONFIG']['ollama_host'] = public_url
                else:
                    logging.error("No public HTTPS tunnels found.")

                # Save the updated configuration
                save_configuration(config)

            # Add more advanced settings questions here as needed

    except Exception as e:
        logging.exception("An unexpected error occurred in handle_advanced_settings:")



def check_and_refresh_ngrok_tunnel(config, retry_interval=10):
    global ngrok_tunnel_info
    while True:
        try:
            if ngrok_tunnel_info is None:
                ngrok_api_key = get_ngrok_api_key()
                tunnels = get_ngrok_tunnels(ngrok_api_key)
                public_url = get_public_url(tunnels)

                if public_url:
                    ngrok_tunnel_info = public_url
                    config['REMOTE_HOST_CONFIG']['ollama_host'] = public_url
                    save_configuration(config)  # Save the updated configuration
                    logging.info(f"Updated ngrok URL in config: {public_url}")
                    return public_url
                else:
                    logging.info("\nNo active ngrok tunnels found.\n")

            else:
                return ngrok_tunnel_info

            # Countdown for retry
            for remaining in range(retry_interval, 0, -1):
                sys.stdout.write(f"\rRetrying to establish ngrok tunnel connection in {remaining} seconds...")
                sys.stdout.flush()
                time.sleep(1)

            sys.stdout.write("\rRetrying to establish ngrok tunnel connection...     \n")
            sys.stdout.flush()

        except Exception as e:
            logging.error(f"Error checking or refreshing ngrok URL: {e}")
            time.sleep(retry_interval)








def main():
    setup_logging()
    config = configparser.ConfigParser()
    config.read('config.ini')
    log_initial_config(config)
    latest_version, version_message = check_latest_version()
    startup_message = generate_startup_message(latest_version, version_message)
    logging.info(startup_message)
    overall_goal = get_input("Please specify your overall goal: ")
    num_alternative_crews = get_input("How many alternative crews do you wish to generate? [3]: ", default='3', validator=validate_positive_int)
    rank_crews = get_input("Do you want the crews to be ranked afterwards? (yes/no) [yes]: ", default='yes', validator=validate_yes_no) in ['yes', 'y']
    llm_endpoint, llm_model = choose_llm_endpoint_and_model(config)
    handle_advanced_settings(config)

    # Check and refresh ngrok connection before proceeding
    if llm_endpoint == 'ollama' and config.getboolean('REMOTE_HOST_CONFIG', 'use_remote_ollama_host', fallback=False):
        ngrok_url = check_and_refresh_ngrok_tunnel(config)
        if not ngrok_url:
            for i in range(10, 0, -1):
                print(f"\rRetrying in {i} seconds...(Press Ctrl+C to abort)", end='', flush=True)
                time.sleep(1)
                ngrok_url = check_and_refresh_ngrok_tunnel(config)
                if ngrok_url:
                    break

        if not ngrok_url:
            print("\nUnable to establish ngrok tunnel. Operation aborted.")
            logging.info("Operation aborted due to failure to establish ngrok tunnel.")
            sys.exit(1)

    # Before starting any operations that require ngrok tunnel, check whether there is still a connection, and check whether the ngrok_url has changed:
    if llm_endpoint == 'ollama' and config.getboolean('REMOTE_HOST_CONFIG', 'use_remote_ollama_host', fallback=False):
        ngrok_url = check_and_refresh_ngrok_tunnel(config)

    # Pull the model using the updated ngrok URL
    if llm_endpoint == 'ollama' and ngrok_url:
        base_url = config['REMOTE_HOST_CONFIG']['ollama_host']
        pull_result = pull_model(llm_model, base_url, verbose=True)
        if pull_result is None or pull_result.get('status') != 'success':
            logging.error(f"Failed to pull model {llm_model}.")
            sys.exit(1)

    # Create an instance of AutoCrew and log the updated configuration
    config_path = os.path.join(os.getcwd(), 'config.ini')
    autocrew = AutoCrew(config_path)
    autocrew.log_config_with_redacted_api_keys()

    # Execute AutoCrew script and handle rankings if applicable
    if run_autocrew_script(num_alternative_crews, overall_goal, rank_crews):
        if rank_crews:
            handle_ranked_crews(overall_goal)
            print_ranking_csv(overall_goal)  # Print the ranking CSV file after ranking is completed
        else:
            # Additional logic if ranking is not performed
            pass
    else:
        logging.error("Autocrew script execution failed.")

if __name__ == "__main__":
    try:
        clear_screen_and_logfile('autocrew.log')
        main()
    except Exception as e:
        logging.exception("Unhandled exception: %s", str(e))
    finally:
        flush_log_handlers()  # Ensure log handlers are flushed at the end of the script
>>>>>>> Stashed changes
