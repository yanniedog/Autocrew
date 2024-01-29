# filename: core.py

# Standard library imports
import argparse
import configparser
import copy
import csv
import datetime
import io
import json
import logging
import os
import re
import requests
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

# External libraries imports
from packaging import version
from openai import OpenAI
from datetime import datetime


# Local application/utility specific imports
from utils import (
    count_tokens, get_next_crew_name, parse_csv_data,
    save_csv_output, write_crewai_script, countdown_timer,
    redact_api_key, GREEK_ALPHABETS
)
from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


    
class AutoCrew():    
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        self.config.read(config_file)

        # BASIC section
        self.llm_endpoint = self.config.get('BASIC', 'llm_endpoint', fallback=None)

        # OLLAMA_CONFIG section
        self.llm_model = self.config.get('OLLAMA_CONFIG', 'llm_model', fallback=None)

        # OPENAI_CONFIG section
        self.openai_model = self.config.get('OPENAI_CONFIG', 'openai_model', fallback=None)
        try:
            self.openai_max_tokens = int(self.config.get('OPENAI_CONFIG', 'max_tokens', fallback=0))
            logging.debug(f"Loaded max_tokens from config: {self.openai_max_tokens}")
        except ValueError:
            logging.error("Invalid value for max_tokens in config file.")
            self.openai_max_tokens = 0

        # CREWAI_SCRIPTS section
        self.llm_endpoint_within_generated_scripts = self.config.get('CREWAI_SCRIPTS', 'llm_endpoint_within_generated_scripts', fallback=None)
        self.llm_model_within_generated_scripts = self.config.get('CREWAI_SCRIPTS', 'llm_model_within_generated_scripts', fallback=None)
        self.add_api_keys_to_crewai_scripts = self.config.getboolean('CREWAI_SCRIPTS', 'add_api_keys_to_crewai_scripts', fallback=False)
        self.add_ollama_host_url_to_crewai_scripts = self.config.getboolean('CREWAI_SCRIPTS', 'add_ollama_host_url_to_crewai_scripts', fallback=False)
        self.overall_goal_truncation_for_filenames = self.config.getint('CREWAI_SCRIPTS', 'overall_goal_truncation_for_filenames', fallback=40)

        # AUTHENTICATORS section
        self.openai_api_key = self.config.get('AUTHENTICATORS', 'openai_api_key', fallback=None)
        self.ngrok_auth_token = self.config.get('AUTHENTICATORS', 'ngrok_auth_token', fallback=None)
        self.ngrok_api_key = self.config.get('AUTHENTICATORS', 'ngrok_api_key', fallback=None)

        # REMOTE_HOST_CONFIG section
        self.reset_ollama_host_on_startup = self.config.getboolean('REMOTE_HOST_CONFIG', 'reset_ollama_host_on_startup', fallback=False)
        self.use_remote_ollama_host = self.config.getboolean('REMOTE_HOST_CONFIG', 'use_remote_ollama_host', fallback=False)
        self.name_of_remote_ollama_host = self.config.get('REMOTE_HOST_CONFIG', 'name_of_remote_ollama_host', fallback=None)

        # MISCELLANEOUS section
        self.on_screen_logging_level = self.config.get('MISCELLANEOUS', 'on_screen_logging_level', fallback='INFO')

        # Set a default value for ollama_host
        self.ollama_host = "http://localhost:11434"  # Default value

        # Initialize other components
        self.ollama = self.initialize_ollama() if self.llm_endpoint == 'ollama' else None
        # self.openai = self.initialize_openai() if self.llm_endpoint == 'openai' else None  (this is not needed, as openai does not need to be initalised like Ollama)


    
    def load_config(self, config_file):
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        config.read(config_file)
        return config
    
    def initialize_ollama(self):
        connection_type = "remote" if self.use_remote_ollama_host else "local"
        model = self.llm_model
        logging.info(f"Initializing {connection_type} connection to Ollama using model {model}...")

        # Start the Ollama service if it's not already running
        self.start_ollama_service()

        # Set default Ollama host if not specified in environment
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        logging.info(f"Ollama host: {self.ollama_host}")

        try:
            return Ollama(base_url=self.ollama_host, model=self.llm_model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        except Exception as e:
            logging.error(f"Failed to initialize Ollama: {e}")
            return None
        
    def start_ollama_service(self):
        try:
            # Check if the Ollama service is running
            subprocess.check_output(["pgrep", "-f", "ollama serve"])
            logging.debug("Ollama service is already running.")
        except subprocess.CalledProcessError:
            # If the Ollama service is not running, start it
            logging.debug("Starting Ollama service...")
            subprocess.Popen(["ollama", "serve"], start_new_session=True)
            
    def is_ollama_running(self):
        try:
            subprocess.check_output(["pgrep", "-f", "ollama serve"])
            logging.debug("Ollama service is already running.")
            return True
        except subprocess.CalledProcessError:
            return False
        
    def get_agent_data(self, overall_goal, delimiter):
        if self.llm_endpoint == 'ollama':
            connection_type = "remote" if self.use_remote_ollama_host else "local"
            model = self.llm_model
            llm_name = f"Ollama using model {model}"
        else:
            connection_type = "remote"
            model = self.openai_model
            llm_name = f"OpenAI using model {model}"

        logging.debug(f"Initializing {connection_type} connection to {llm_name} for generating agent data with the overall goal of '{overall_goal}'...")
        
        # Construct the instruction including the overall_goal
        instruction = (
            f'Create a dataset in a CSV format with each field enclosed in double quotes, '
            f'for a team of agents with the goal: "{overall_goal}". '
            f'Use the delimiter "{delimiter}" to separate the fields. '
            'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
            'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
            'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
        )
        
        # Calculate the number of tokens in the complete instruction
        instruction_tokens = count_tokens(instruction)
        logging.debug(f"Instruction given to {llm_name}:\n{instruction}")
        logging.debug(f"Number of tokens in the instruction: {instruction_tokens}")

        # Read the max_tokens parameter from config.ini
        max_tokens = self.openai_max_tokens
        logging.debug(f"Max tokens from config: {max_tokens}")

        # Subtract the number of tokens in the instruction from max_tokens
        max_response_tokens = max_tokens - instruction_tokens - 200
        logging.debug(f"Max response tokens available for LLM response: {max_response_tokens}")

        # Ensure max_response_tokens is not negative
        if max_response_tokens < 0:
            logging.error("The number of tokens in the instruction exceeds the max_tokens limit.")
            max_response_tokens = 0

        try:
            if self.llm_endpoint == 'ollama' and self.ollama:
                response = self.ollama.invoke(instruction)
                # Log the raw LLM output
                logging.debug(f"Raw LLM output (Ollama):\n{response}")
                logging.debug(f"Number of tokens in the response: {count_tokens(response)}")
                return response
            elif self.llm_endpoint == 'openai' and self.openai_api_key:
                client = OpenAI(api_key=self.openai_api_key)
                chat_completion = client.chat.completions.create(
                    model=self.openai_model,  # Use the model directly from the configuration
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": instruction}
                    ],
                    max_tokens=max_response_tokens  # Use the calculated max_response_tokens
                )
                response = chat_completion.choices[0].message.content.strip()
                # Log the raw LLM output
                logging.debug(f"Raw LLM output (OpenAI):\n{response}")
                logging.debug(f"Number of tokens in the response: {count_tokens(response)}")
                return response
            else:
                logging.error("Neither OpenAI API key nor Ollama instance is available.")
                return ""
        except Exception as e:
            logging.error(f"Error in API call: {e}")
            return ""

    def generate_scripts(self, overall_goal, num_scripts):
        csv_file_paths = []
        for i in range(num_scripts):
            crew_name = get_next_crew_name(overall_goal)  # Get the next available crew name
            logging.info(f"\nGenerating crew {i + 1} of {num_scripts} ('{crew_name}' crew)...")
            file_path = self.generate_single_script(i, num_scripts, overall_goal, crew_name)
            csv_file_paths.append(file_path)
        return csv_file_paths

        
    # Define a function to process LLM response
    def generate_single_script(self, i, num_scripts, overall_goal, crew_name):
        def process_response(response):
            # Determine the Greek letter suffix for this crew
            greek_suffix = get_next_crew_name(overall_goal)

            # Pass the truncation length to the save_csv_output function
            file_path = save_csv_output(response, overall_goal, truncation_length=self.overall_goal_truncation_for_filenames, greek_suffix=greek_suffix)
            agents_data = parse_csv_data(response, delimiter=',', filename=file_path)
            if not agents_data:
                raise ValueError('No agent data parsed')
            # Use the truncated goal for the script filename
            truncated_goal = overall_goal[:self.overall_goal_truncation_for_filenames].replace(" ", "-")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f'crewai-autocrew-{timestamp}-{truncated_goal}-{greek_suffix}.py'
            
            # Generate crew tasks based on the agents_data
            crew_tasks = self.generate_crew_tasks(agents_data)
            
            # Call the standalone function with the necessary parameters
            write_crewai_script(
                agents_data,
                crew_tasks,
                file_name,
                self.llm_endpoint_within_generated_scripts,
                self.llm_model_within_generated_scripts,
                self.add_ollama_host_url_to_crewai_scripts,
                self.ollama_host,
                self.add_api_keys_to_crewai_scripts,
                self.openai_api_key,
                self.openai_model
            )
            return file_path

        # Construct the instruction including the overall_goal
        instruction = (
            f'Create a dataset in a CSV format with each field enclosed in double quotes, '
            f'for a team of agents with the goal: "{overall_goal}". '
            f'Use the delimiter "," to separate the fields. '
            'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
            'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
            'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
        )

        # Call the LLM with retry logic
        return self.call_llm_with_retry(instruction, overall_goal, process_response)


 
    def call_llm_with_retry(self, instruction, overall_goal, process_response_func):
        max_attempts = 3
        for attempt in range(max_attempts):
            logging.debug(f"LLM call attempt {attempt + 1} for the goal: '{overall_goal}'")
            response = self.get_agent_data(instruction, ',')
            if not response:
                logging.error('No response from LLM')
                if attempt == max_attempts - 1:
                    raise ValueError("Failed to get valid response from LLM after 3 attempts.")
                continue

            try:
                return process_response_func(response)
            except ValueError as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise ValueError("Failed to process LLM response after 3 attempts.")
                

    def rank_crews(self, csv_file_paths, overall_goal, verbose=False):
        logging.info("Starting the ranking process...")  # Add this line to confirm the method is called
    
        ranked_crews = []
        overall_summary = ""

        # Determine connection type and model for LLM
        if self.llm_endpoint == 'ollama':
            connection_type = "remote" if self.use_remote_ollama_host else "local"
            model = self.llm_model
            llm_name = f"Ollama using model {model}"
        else:
            connection_type = "remote"
            model = self.openai_model
            llm_name = f"OpenAI using model {model}"

        logging.info(f"Initializing {connection_type} connection to {llm_name} for ranking crews with the overall goal of '{overall_goal}'...")

        concatenated_csv_data = 'crew_name,role,goal,backstory,assigned_task,allow_delegation\n'
        for file_path in csv_file_paths:
            try:
                # Extract the Greek letter (crew name) from the filename
                crew_name = os.path.basename(file_path).split('-')[-1].split('.')[0]
                # Check if the crew name is a valid Greek letter
                if crew_name.lower() not in GREEK_ALPHABETS:
                    logging.debug(f"Skipping file {file_path} as it does not end with a Greek letter.")
                    continue
                with open(file_path, 'r') as file:
                    csv_data = file.read().strip()
                if csv_data.count('\n') < 1:
                    continue
                # Skip the first line (the remark) and add the crew name to each line
                csv_lines = csv_data.split('\n')[1:]
                csv_lines_with_crew_name = [f'"{crew_name}",' + line for line in csv_lines]
                concatenated_csv_data += '\n'.join(csv_lines_with_crew_name) + '\n'
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

        # Log the concatenated CSV data
        logging.debug(f"Concatenated CSV Data:\n{concatenated_csv_data}")

        # Log the token count for the concatenated CSV data
        concatenated_csv_token_count = count_tokens(concatenated_csv_data)
        logging.debug(f"Token count for concatenated CSV data: {concatenated_csv_token_count}")

        # Prepare the ranking prompt
        csv_reader = csv.DictReader(io.StringIO(concatenated_csv_data))
        json_data = [row for row in csv_reader]
        json_data_str = json.dumps(json_data)
        if verbose:
            logging.debug('\nConcatenated CSV Data:')
            logging.debug(concatenated_csv_data)

        crew_names_str = ', '.join([os.path.basename(file_path).split('-')[-1].split('.')[0] for file_path in csv_file_paths if os.path.basename(file_path).split('-')[-1].split('.')[0].lower() in GREEK_ALPHABETS])
        prompt = (
            f"Analyze the following list of crews ({crew_names_str}) to determine their suitability for successfully completing the task: "
            f"{overall_goal}. The crews are represented in a JSON object format: {json_data_str}. "
            "Please provide a ranking of the crews by their names, with the most suitable crew listed first. "
            "Also, provide a brief critique for each crew, highlighting their strengths and weaknesses."
        )

        # Log the entire ranking request
        logging.debug(f"Ranking request:\n{prompt}")

        # Log the token count for the entire ranking request
        prompt_token_count = count_tokens(prompt)
        logging.debug(f"Token count for the entire ranking request: {prompt_token_count}")

        # Calculate max_response_tokens
        max_response_tokens = self.openai_max_tokens - prompt_token_count - 200
        if max_response_tokens < 0:
            logging.error("The number of tokens in the prompt exceeds the max_tokens limit.")
            max_response_tokens = 0

        # Log the calculated max_response_tokens
        logging.debug(f"Max response tokens available for LLM response: {max_response_tokens}")

        # Make the API call and process the response
        if self.llm_endpoint == 'ollama' and self.ollama:
            ranked_crew = self.ollama.invoke(prompt)
        elif self.llm_endpoint == 'openai' and self.openai_api_key:
            client = OpenAI(api_key=self.openai_api_key)
            chat_completion = client.chat.completions.create(model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_response_tokens)
            ranked_crew = chat_completion.choices[0].message.content.strip()
        else:
            logging.error("Neither OpenAI API key nor Ollama instance is available.")
            return []

        logging.debug(f"Number of tokens in the response: {count_tokens(ranked_crew)}")

        ranked_crews.append((concatenated_csv_data, ranked_crew))
        overall_summary += f'\n\nCrews in the following CSV files were ranked:\n'
        for file_path in csv_file_paths:
            overall_summary += f'{file_path}\n'
        overall_summary += f'\nRanking Summary:\n{ranked_crew}'

        return ranked_crews, overall_summary
        logging.info("Ranking process completed.")  # Add this line to confirm the method has finished
    def get_existing_scripts(self, overall_goal):
        # Assuming scripts are stored in a directory named "scripts"
        script_dir = os.path.join(os.getcwd(), "scripts")
        return [os.path.join(script_dir, f) for f in os.listdir(script_dir) if f.endswith('.csv') and overall_goal[:40] in f]
    
    

                
    def save_ranking_output(self, ranked_crews, overall_goal):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f'crewai-autocrew-{timestamp}-{overall_goal[:40].replace(" ", "-")}-ranking.csv'
        directory = os.path.join(os.getcwd(), "scripts")
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["crew_name", "ranking"])
            for crew_name, ranking in ranked_crews:
                writer.writerow([crew_name, ranking])
        logging.info(f"\nYour crews have been ranked successfully.\nSee here for details: {file_path}\n")  # Log the full path of the saved ranking CSV
    
    def log_config_with_redacted_api_keys(self):
        # Create a copy of the config to redact API keys
        redacted_config = copy.deepcopy(self.config)
        for section in redacted_config.sections():
            for key, value in redacted_config.items(section):
                if 'api_key' in key.lower() or 'auth_token' in key.lower():
                    # Redact all but the last 4 characters of the API key
                    redacted_config.set(section, key, '*' * (len(value) - 4) + value[-4:])
        
        # Convert the redacted config to a string
        config_string = io.StringIO()
        redacted_config.write(config_string)
        config_string.seek(0)  # Reset the StringIO object to the beginning

        # Log the redacted config
        logging.debug("Redacted config.ini content:\n" + config_string.read())    
        

        
    def check_latest_version(self, current_version):
        try:
            response = requests.get('https://api.github.com/repos/yanniedog/autocrew/releases/latest')
            response.raise_for_status()
            latest_release = response.json()
            latest_version = latest_release['tag_name']

            if version.parse(latest_version) > version.parse(current_version):
                logging.info(f"An updated version of AutoCrew is available: {latest_version}")
                logging.info("Consider updating to the latest version for new features and bug fixes.")
                logging.info(f"Release notes: {latest_release['body']}")
                logging.info(f"Download the latest version here: {latest_release['html_url']}")
            else:
                logging.info("You are running the latest version of AutoCrew.")
        except Exception as e:
            logging.error(f"Error checking for the latest version: {e}")
            
    def run(self, overall_goal, num_scripts, auto_run, verbose):
        if num_scripts is None:
            num_scripts = 1  # Default value if not provided
        csv_file_paths = self.generate_scripts(overall_goal, num_scripts)
        if auto_run:
            for path in csv_file_paths:
                script_path = path.replace('.csv', '.py')  # Change the file extension to .py
                subprocess.run([sys.executable, script_path])  # Using sys.executable
                
    def generate_crew_tasks(self, agents_data):
        return [{'role': agent['role']} for agent in agents_data]

    
    def get_task_var_name(self, role):
        return f'task_{role.replace(" ", "_").replace("-", "_").replace(".", "_")}'
            



