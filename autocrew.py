# Autocrew v2.0.1 (2024-01-26) 
# https://github.com/yanniedog/autocrew

import argparse
import configparser
import csv
import io
import json
import logging
import os
import requests
import subprocess
import sys
import tiktoken
import time
from datetime import datetime
from packaging import version
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from openai import OpenAI

# Autocrew version
AUTOCREW_VERSION = "1.5"

GREEK_ALPHABETS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
                       "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]


def install_dependencies():
    if not os.path.exists('requirements.txt'):
        raise FileNotFoundError("requirements.txt not found.")
    print("Installing dependencies...")
    subprocess.run(['pip', 'install', '--requirement', 'requirements.txt'])
    print("Dependencies installed.")
    
class AutoCrew:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        self.config.read(config_file)

        # BASIC section
        self.llm_endpoint = self.config.get('BASIC', 'llm_endpoint', fallback=None)
        self.llm_model = self.config.get('BASIC', 'llm_model', fallback=None)

        # OPENAI_CONFIG section
        self.openai_model = self.config.get('OPENAI_CONFIG', 'openai_model', fallback=None)
        self.openai_engine = self.config.get('OPENAI_CONFIG', 'openai_engine', fallback=None)
        try:
            self.openai_max_tokens = int(self.config.get('OPENAI_CONFIG', 'openai_max_tokens', fallback=0))
        except ValueError:
            logging.error("Invalid value for openai_max_tokens in config file.")
            self.openai_max_tokens = 0

        # CREWAI_SCRIPTS section
        self.llm_endpoint_within_generated_scripts = self.config['CREWAI_SCRIPTS']['llm_endpoint_within_generated_scripts']
        self.llm_model_within_generated_scripts = self.config['CREWAI_SCRIPTS']['llm_model_within_generated_scripts']
        self.add_api_keys_to_crewai_scripts = self.config['CREWAI_SCRIPTS']['add_api_keys_to_crewai_scripts'] == 'y'
        self.add_ollama_host_url_to_crewai_scripts = self.config['CREWAI_SCRIPTS']['add_ollama_host_url_to_crewai_scripts'] == 'y'

        # AUTHENTICATORS section
        self.openai_api_key = self.config['AUTHENTICATORS']['openai_api_key']
        self.ngrok_auth_token = self.config['AUTHENTICATORS']['ngrok_auth_token']
        self.ngrok_api_key = self.config['AUTHENTICATORS']['ngrok_api_key']

        # REMOTE_HOST_CONFIG section
        self.reset_ollama_host_on_startup = self.config['REMOTE_HOST_CONFIG']['reset_ollama_host_on_startup'] == 'y'
        self.use_remote_ollama_host = self.config['REMOTE_HOST_CONFIG']['use_remote_ollama_host'] == 'y'
        self.name_of_remote_ollama_host = self.config['REMOTE_HOST_CONFIG']['name_of_remote_ollama_host']

        # Initialize other components
        self.initialize_logging()
        self.ollama = self.initialize_ollama()  # Always initialize the Ollama instance
        
    def initialize_logging(self):
        log_file = os.path.join(os.getcwd(), 'autocrew.log')
        logging.basicConfig(
            filename=log_file,
            filemode='w',  # Overwrite the log file
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        # Add a stream handler to also output to console
        console_handler = logging.StreamHandler()
        on_screen_logging_level = self.config.get('MISCELLANEOUS', 'on_screen_logging_level', fallback='WARNING')
        console_handler.setLevel(getattr(logging, on_screen_logging_level.upper()))
        logging.getLogger().addHandler(console_handler)
    
    def countdown_timer(seconds: int):
        """Displays a countdown timer for the specified number of seconds."""
        for i in range(seconds, 0, -1):
            print(f"Pausing for {i} seconds")
            time.sleep(1)
        print("Continuing...") 

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
            logging.info("Ollama service is already running.")
        except subprocess.CalledProcessError:
            # If the Ollama service is not running, start it
            logging.info("Starting Ollama service...")
            subprocess.Popen(["ollama", "serve"], start_new_session=True)



    @staticmethod
    def count_tokens(string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding_name = 'cl100k_base'  # Assuming this is the encoding you want to use
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    


    def get_agent_data(self, overall_goal, delimiter):
        if self.llm_endpoint == 'ollama':
            connection_type = "remote" if self.use_remote_ollama_host == 'y' else "local"
            model = self.llm_model
            llm_name = f"Ollama using model {model}"
        else:
            connection_type = "remote"
            model = self.openai_engine
            llm_name = f"OpenAI using engine {model}"

        print(f"Initializing {connection_type} connection to {llm_name} in order to generate agent data with the overall goal of '{overall_goal}'...")
        instruction = (
            f'Create a dataset in a CSV format with each field enclosed in double quotes, '
            f'for a team of agents with the goal: "{overall_goal}". '
            f'Use the delimiter "{delimiter}" to separate the fields. '
            'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
            'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
            'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
        )
        
        print()
        print(f"Instruction given to {llm_name}:")
        print()
        print(f"{instruction}")
        print()
        print(f"Number of tokens in the instruction: {self.count_tokens(instruction)}")
        
        try:
            if self.llm_endpoint == 'ollama' and self.ollama:
                response = self.ollama.invoke(instruction)
                # Log the raw LLM output
                logging.info(f"Raw LLM output (Ollama):\n{response}")
                print(f"Number of tokens in the response: {self.count_tokens(response)}")
                return response
            elif self.llm_endpoint == 'openai' and self.openai_api_key:
                client = OpenAI(api_key=self.openai_api_key)
                chat_completion = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": instruction}
                    ],
                    max_tokens=self.openai_max_tokens
                )
                response = chat_completion.choices[0].message.content.strip()
                # Log the raw LLM output
                logging.info(f"Raw LLM output (OpenAI):\n{response}")
                print(f"Number of tokens in the response: {self.count_tokens(response)}")
                return response
            else:
                logging.error("Neither OpenAI API key nor Ollama instance is available.")
                return ""
        except Exception as e:
            logging.error(f"Error in API call: {e}")
            return ""


    def get_next_crew_name(self, overall_goal):
        # Define the target directory as 'generated' within the current working directory
        target_directory = os.path.join(os.getcwd(), "scripts")
        
        # Check if the target directory exists, create it if not
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        formatted_goal = overall_goal.replace(" ", "-")
        existing_files = [f for f in os.listdir(target_directory) if (f.endswith('.csv') or f.endswith('.py')) and formatted_goal in f]
        existing_indices = []

        for file_name in existing_files:
            name_parts = file_name.split('-')
            if len(name_parts) > 3:  # Ensure the filename has enough parts
                alpha = name_parts[-1].split('.')[0]  # Get the part before file extension
                if alpha in GREEK_ALPHABETS:
                    existing_indices.append(GREEK_ALPHABETS.index(alpha))

        if existing_indices:
            next_index = max(existing_indices) + 1
        else:
            next_index = 0

        return GREEK_ALPHABETS[next_index % len(GREEK_ALPHABETS)]

    def save_csv_output(self, response, overall_goal):
        # Log the initial raw CSV data
        logging.info(f"Initial raw CSV data:\n{response}")

        # Extract the CSV data between the ``` marks
        csv_data = response.split('```')[1] if '```' in response else response
        csv_data = csv_data.strip()
        logging.info("Extracted CSV data from raw output.")

        # Define the correct header
        correct_header = ['crew_name', 'role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']

        # Split the CSV data into lines
        lines = csv_data.split('\n')

        # Split the first line (header) into values
        header_values = lines[0].split(',')

        # Check each header value and replace any incorrect value with the correct one
        for i, value in enumerate(header_values):
            if value != correct_header[i]:
                header_values[i] = correct_header[i]
                logging.info(f"Corrected header value: {value} to {correct_header[i]}")

        # Join the header values back into a string
        lines[0] = ','.join(header_values)
        logging.info("Corrected header values and joined back into a string.")

        # Join the lines back into a string
        csv_data = '\n'.join(lines)
        logging.info("Joined lines back into a single CSV string.")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        crew_name = self.get_next_crew_name(overall_goal)  # Correctly get the next crew name
        clean_crew_name = crew_name.strip('"')  
        file_name = f'crewai-autocrew-{timestamp}-{overall_goal[:40].replace(" ", "-")}-{clean_crew_name}.csv'
        directory = os.path.join(os.getcwd(), "scripts")
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as file:
            file.write(lines[0] + '\n')  # Write the header
            for line in lines[1:]:
                if line.strip():
                    modified_line = f'"{crew_name}",{line}\n'
                    file.write(modified_line)
        logging.info(f"CSV file saved at: {file_path}")  # Log the path of the saved file
        return file_path

    def parse_csv_data(self, response, delimiter=',', filename=''):
        # Log the raw CSV data before parsing
        logging.info(f"Raw CSV data for parsing:\n{response}")

        header = ['role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
        agents_data = []
        csv_data = csv.reader(io.StringIO(response), delimiter=delimiter)
        lines = list(csv_data)

        if not lines:
            logging.error("CSV data is empty after splitting into lines.")
            raise ValueError('CSV data is empty')

        header_line = lines[0]
        header_mapping = {h.lower(): h for h in header}
        header_indices = [header_mapping.get(h.lower()) for h in header_line]

        if not all(header_indices):
            logging.error('Header component missing or incorrect in CSV data')
            raise ValueError('Header component missing or incorrect in CSV data')

        logging.info(f"Parsed header: {header_line}")

        for line in lines[1:]:
            agent_data = {}
            for i, value in enumerate(line):
                header_name = header_indices[i]
                if header_name:
                    agent_data[header_name] = value.strip('"').strip()
            if 'role' not in agent_data or not agent_data['role']:
                logging.error('Role component missing in line of CSV data')
                raise ValueError('Role component missing in CSV data')
            agent_data['filename'] = filename
            agents_data.append(agent_data)

        logging.info(f"Successfully parsed {len(agents_data)} agents from CSV data.")
        return agents_data

    def define_agent(self, agent, search_tool, llm):
        role_var = agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_').replace(' ', '')
        role_value = agent['role'].replace('"', '\\"').replace("'", "\\'")
        backstory = agent['backstory'].replace('"', '\\"').replace("'", "\\'")
        delegation = 'True' if agent['allow_delegation'] == 'True' else 'False'
        return (
            f'{role_var} = Agent(\n'
            f'    role="{role_value}",\n'
            f'    goal="{agent["goal"]}",\n'
            f'    backstory="{backstory}",\n'
            f'    verbose=True,\n'
            f'    allow_delegation={delegation},\n'
            f'    llm={llm},\n'  # <----- passing our llm reference here
            f'    tools=[{search_tool}]\n'
            ')\n\n'
        )

    def get_task_var_name(self, role):
        return f'task_{role.replace(" ", "_").replace("-", "_").replace(".", "_")}'

    def define_task(self, agent):
        task_var = self.get_task_var_name(agent['role'])
        task_description = agent["assigned_task"].strip().replace('"', '\\"')
        return (
            f'{task_var} = Task(\n'
            f' description="{task_description}",\n'
            f' agent={agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")},\n'
            ' verbose=True,\n'
            ')\n\n'
        )

    def generate_crew_tasks(self, agents_data):
        return ', '.join([self.get_task_var_name(agent["role"]) for agent in agents_data])

    def write_crewai_script(self, agents_data, crew_tasks, file_name):
        # Implementation to handle agents_data, crew_tasks, and file_name
        pass
        crew_agents = ', '.join([agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_') for agent in agents_data])
        with open(os.path.join("scripts", file_name), 'w') as file:
            # Script header and imports
            file.write(
                'import os\n'
                'from langchain_community.llms import Ollama\n'
                'from langchain_community.tools import DuckDuckGoSearchRun\n'
                'from crewai import Agent, Task, Crew, Process\n'
                'import openai\n\n'
            )

            # Check and write LLM configuration based on settings
            if self.llm_endpoint_within_generated_scripts == 'ollama':
                if self.add_ollama_host_url_to_crewai_scripts:
                    file.write(f'ollama_host = "{self.ollama_host}"\n')  
                file.write(f'ollama = Ollama(model="{self.llm_model_within_generated_scripts}", base_url=\'{self.ollama_host if self.add_ollama_host_url_to_crewai_scripts else ""}\')\n')
            elif self.llm_endpoint_within_generated_scripts == 'openai':
                if self.add_api_keys_to_crewai_scripts:
                    file.write(f'os.environ["OPENAI_API_KEY"] = "{self.openai_api_key}"\n')
                file.write(f'llm = openai.chatcompletion.create(model="{self.openai_model}")\n')

            # Other script content
            file.write('search_tool = DuckDuckGoSearchRun()\n\n')

            # Define agents and their tasks
            for agent in agents_data:
                agent_var = self.define_agent(agent, "search_tool", 'ollama' if self.llm_endpoint_within_generated_scripts == 'ollama' else 'openai')
                file.write(agent_var + '\n')
                task_var = self.define_task(agent)
                file.write(task_var + '\n')

            # Define crew
            crew_tasks = ', '.join([self.get_task_var_name(agent["role"]) for agent in agents_data])
            file.write(
                'crew = Crew(\n'
                f'    agents=[{crew_agents}],\n'
                f'    tasks=[{crew_tasks}],\n'
                '    verbose=True,\n'
                '    process=Process.sequential,\n'
                ')\n\n'
                '# Kickoff the crew tasks\n'
                'result = crew.kickoff()\n\n'
                '# Handle the "result" as needed\n'
            )
        print(f"Script saved at: {os.path.join('scripts', file_name)}")  # Print the full path of the saved file

    
    def call_llm_with_retry(self, instruction, overall_goal, process_response_func):
        max_attempts = 3
        for attempt in range(max_attempts):
            logging.info(f"LLM call attempt {attempt + 1} for the goal: '{overall_goal}'")
            response = self.get_agent_data(instruction, ',')  # Assuming get_agent_data makes the actual LLM call
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
    def generate_single_script(self, i, num_scripts, overall_goal):
        # Define a function to process LLM response
        def process_response(response):
            file_path = self.save_csv_output(response, overall_goal)
            agents_data = self.parse_csv_data(response, delimiter=',', filename=file_path)
            if not agents_data:
                raise ValueError('No agent data parsed')
            file_name = os.path.basename(file_path).replace('.csv', '.py')
            crew_tasks = self.generate_crew_tasks(agents_data)
            self.write_crewai_script(agents_data, crew_tasks, file_name)
            return file_path

        instruction = (
            f'Create a dataset in a CSV format with each field enclosed in double quotes, '
            f'for a team of agents with the goal: "{overall_goal}". '
            f'Use the delimiter "," to separate the fields. '
            'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
            'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
            'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
        )

        return self.call_llm_with_retry(instruction, overall_goal, process_response)

    def generate_scripts(self, overall_goal, num_scripts):
        csv_file_paths = []
        for i in range(num_scripts):
            print(f"Generating crew {i + 1} of {num_scripts} with the overall goal of '{overall_goal}'...")
            file_path = self.generate_single_script(i, num_scripts, overall_goal)
            csv_file_paths.append(file_path)
        return csv_file_paths


    def rank_crews(self, csv_file_paths, overall_goal, verbose=False):
        ranked_crews = []
        overall_summary = ""

        # Determine connection type and model for LLM
        if self.llm_endpoint == 'ollama':
            connection_type = "remote" if self.use_remote_ollama_host else "local"
            model = self.llm_model
            llm_name = f"Ollama using model {model}"
        else:
            connection_type = "remote"
            model = self.openai_engine
            llm_name = f"OpenAI using engine {model}"

        print(f"Initializing {connection_type} connection to {llm_name} for ranking crews with the overall goal of '{overall_goal}'...")

        concatenated_csv_data = 'crew_name,role,goal,backstory,assigned_task,allow_delegation\n'
        for file_path in csv_file_paths:
            try:
                with open(file_path, 'r') as file:
                    csv_data = file.read().strip()
                if csv_data.count('\n') < 1:
                    continue
                concatenated_csv_data += csv_data[csv_data.index('\n') + 1:] + '\n'
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

        if concatenated_csv_data.strip() == 'crew_name,role,goal,backstory,assigned_task,allow_delegation':
            logging.warning("No valid data found in the provided CSV files.")
            return [], "No ranking could be performed due to insufficient data."

        csv_reader = csv.DictReader(io.StringIO(concatenated_csv_data))
        json_data = [row for row in csv_reader]
        json_data_str = json.dumps(json_data)
        if verbose:
            logging.debug('\nConcatenated CSV Data:')
            logging.debug(concatenated_csv_data)

        crew_names_str = ', '.join([os.path.basename(file_path).split('-')[-1].split('.')[0] for file_path in csv_file_paths])
        prompt = (
            f"Analyze the following list of crews ({crew_names_str}) to determine their suitability for successfully completing the task: "
            f"{overall_goal}. The crews are represented in a JSON object format: {json_data_str}. "
            "Please provide a ranking of the crews by their names, with the most suitable crew listed first. "
            "Also, provide a brief critique for each crew, highlighting their strengths and weaknesses."
        )

        num_tokens = self.count_tokens(prompt)
        if num_tokens > 10000:
            raise ValueError(f"The prompt is too long ({num_tokens} tokens). It must be less than 10,000 tokens.")
        if verbose:
            print(f"Number of tokens in the prompt: {num_tokens}")

        if self.llm_endpoint == 'ollama' and self.ollama:
            ranked_crew = self.ollama.invoke(prompt)
        elif self.llm_endpoint == 'openai' and self.openai_api_key:
            client = OpenAI(api_key=self.openai_api_key)
            chat_completion = client.chat.completions.create(model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.openai_max_tokens)
            ranked_crew = chat_completion.choices[0].message.content.strip()
        else:
            logging.error("Neither OpenAI API key nor Ollama instance is available.")
            return []

        print(f"Number of tokens in the response: {self.count_tokens(ranked_crew)}")


        ranked_crews.append((concatenated_csv_data, ranked_crew))
        overall_summary += f'\n\nCrews in the following CSV files were ranked:\n'
        for file_path in csv_file_paths:
            overall_summary += f'{file_path}\n'
        overall_summary += f'\nRanking Summary:\n{ranked_crew}'

        return ranked_crews, overall_summary


    def run(self, overall_goal, num_scripts, auto_run, verbose):
        if num_scripts is None:
            num_scripts = 1  # Default value if not provided
        csv_file_paths = self.generate_scripts(overall_goal, num_scripts)
        if auto_run:
            for path in csv_file_paths:
                script_path = path.replace('.csv', '.py')  # Change the file extension to .py
                subprocess.run([sys.executable, script_path])  # Using sys.executable

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
        print(f"Ranking CSV saved at: {file_path}")  # Print the full path of the saved ranking CSV

               
    

def main():
    parser = argparse.ArgumentParser(description='CrewAI Autocrew Script')
    parser.add_argument('overall_goal', nargs='?', type=str, help='The overall goal for the crew')
    parser.add_argument('-m', '--multiple', type=int, nargs='?', const=-1, default=None, metavar='NUM', help='Create NUM number of CrewAI scripts for the same overall goal. Example: -m 3')
    parser.add_argument('-r', '--rank', action='store_true', help='Rank the generated crews if multiple scripts are created')
    parser.add_argument('-a', '--auto_run', action='store_true', help='Automatically run the scripts after generation')
    parser.add_argument('-v', '--verbose', action='store_true', help='Provide additional details during execution')

    args = parser.parse_args()

    if args.overall_goal is None:
        args.overall_goal = input("Please set the overall goal for your crew: ")

    if args.multiple == -1:
        while True:
            try:
                args.multiple = int(input("Please specify the total number of alternative scripts to generate: "))
                if args.multiple > 0:
                    break
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    autocrew = AutoCrew()

    try:
        if args.multiple:
            csv_file_paths = autocrew.generate_scripts(args.overall_goal, args.multiple)
            if args.rank:
                ranked_crews, overall_summary = autocrew.rank_crews(csv_file_paths, args.overall_goal, args.verbose)
                print(f"\nRanking prompt:\n{overall_summary}\n")
                logging.info(overall_summary)
                autocrew.save_ranking_output(ranked_crews, args.overall_goal)
            if args.auto_run:
                subprocess.run([sys.executable, csv_file_paths[0]])
        elif args.rank:
            csv_file_paths = autocrew.get_existing_scripts(args.overall_goal)
            ranked_crews, overall_summary = autocrew.rank_crews(csv_file_paths, args.overall_goal, args.verbose)
            print(f"\nRanking prompt:\n{overall_summary}\n")
            logging.info(overall_summary)
            autocrew.save_ranking_output(ranked_crews, args.overall_goal)
            if args.auto_run:
                subprocess.run([sys.executable, ranked_crews[0][0]])
        else:
            csv_file_path = autocrew.run(args.overall_goal, None, args.auto_run, args.verbose)
            if args.auto_run:
                subprocess.run([sys.executable, csv_file_path])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    main()
