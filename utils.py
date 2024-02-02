# filename: utils.py

#import csv
import io
import logging
import os
import re
import time
import tiktoken
import json
from textwrap import dedent
from datetime import datetime



GREEK_ALPHABETS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon"
]

def count_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = 'cl100k_base'  # Assuming this is the encoding you want to use
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_next_crew_name(overall_goal, script_directory="scripts"):
    """Determines the next crew name based on existing files."""
    directory = os.path.join(os.getcwd(), script_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist

    formatted_goal = overall_goal.replace(" ", "-")
    existing_files = [f for f in os.listdir(directory) if (f.endswith('.json') or f.endswith('.py')) and formatted_goal in f]
    existing_crew_names = [f.split('-')[-1].split('.')[0] for f in existing_files]
    existing_crew_indices = [GREEK_ALPHABETS.index(name) for name in existing_crew_names if name in GREEK_ALPHABETS]

    # Find the next available Greek alphabet name
    for i, name in enumerate(GREEK_ALPHABETS):
        if i not in existing_crew_indices:
            return name

    # If all names are taken, append a number to the last Greek alphabet name
    return f"{GREEK_ALPHABETS[-1]}_{len(existing_crew_indices) + 1}"



def parse_json_data(response, delimiter=',', filename=''):
    """
    Parses Json data from a string response.

    Args:
        response (str): The response string containing json data.
        delimiter (str, optional): The delimiter used in the json data. Defaults to ','.
        filename (str, optional): The filename for reference in logging. Defaults to ''.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing parsed agent data.

    Raises:
        ValueError: If the json data is not found, incomplete, or incorrectly formatted.
    """
    try:
        # Load and parse the JSON
        json_data = json.loads(response)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None


def read_json_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            extracted_json = json.load(file)
        return extracted_json
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None
def save_json_output(json_response, overall_goal, script_directory="scripts", truncation_length=40, greek_suffix=None):
    """Saves the Json output to a file."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use the provided greek_suffix if available, otherwise get the next one
    greek_suffix = greek_suffix or get_next_crew_name(overall_goal, script_directory)
    # Truncate the overall_goal to the specified number of characters for the filename
    truncated_goal = overall_goal[:truncation_length].replace(" ", "-")
    file_name = f'crewai-autocrew-{timestamp}-{truncated_goal}-{greek_suffix}.json'
    directory = os.path.join(os.getcwd(), script_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    
    with open(file_path, 'w') as file:
        json.dump(json_response, file, indent=2)        
        logging.debug(f"json file saved at: {file_path}")

    return file_path

def extract_json_from_placeholder(placeholder_text):
    # Define a regular expression to find JSON content within triple backticks
    json_pattern = re.compile(r'```json(.*?)```', re.DOTALL)

    # Find all JSON matches in the text
    json_matches = json_pattern.findall(placeholder_text)

    # Process each JSON match
    extracted_json = []
    for json_match in json_matches:
        try:
            # Load and parse the JSON
            json_data = json.loads(json_match.strip())  # strip() is used to remove leading/trailing white spaces
            extracted_json.append(json_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    return extracted_json
    


def countdown_timer(seconds: int):
    """Displays a countdown timer for the specified number of seconds."""
    for i in range(seconds, 0, -1):
        logging.debug(f"Pausing for {i} seconds")
        time.sleep(1)
    logging.info("Continuing...")

def redact_api_key(api_key):
    """Redacts all but the last 4 characters of the API key."""
    return '*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else api_key

def write_crewai_script(agents_json, file_name, llm_endpoint_within_generated_scripts, 
                        llm_model_within_generated_scripts, add_ollama_host_url_to_crewai_scripts, 
                        ollama_host, add_api_keys_to_crewai_scripts, openai_api_key, openai_model):
    """
    Generates and writes a CrewAI script based on provided data.

    Args:
        agents_data (list): List of dictionaries containing agent data.
        crew_tasks (list): List of crew tasks.
        file_name (str): Name of the file to be written.
        llm_endpoint_within_generated_scripts (str): LLM endpoint setting.
        llm_model_within_generated_scripts (str): LLM model setting.
        add_ollama_host_url_to_crewai_scripts (bool): Flag to add Ollama host URL.
        ollama_host (str): Ollama host URL.
        add_api_keys_to_crewai_scripts (bool): Flag to add API keys in the script.
        openai_api_key (str): OpenAI API key.
        openai_model (str): OpenAI model setting.
    """
    # Extract a list of roles
    try:
        crew_tasks = [agent['role'] for agent in agents_json[0]]
    except KeyError:
        raise ValueError('Role component missing in json data')

    script_directory = "scripts"
    script_file_path = os.path.join(script_directory, file_name)

    # Create the scripts directory if it doesn't exist
    if not os.path.exists(script_directory):
        os.makedirs(script_directory)

    try:
        with open(script_file_path, 'w') as file:
            # Start writing the script content
            write_script_header(file)
            write_llm_configuration(file, llm_endpoint_within_generated_scripts, llm_model_within_generated_scripts,
                                    add_ollama_host_url_to_crewai_scripts, ollama_host, add_api_keys_to_crewai_scripts,
                                    openai_api_key, openai_model)

            # Define agents and their tasks
            task_vars, crew_agents = write_agents_and_tasks(file, agents_json)

            # Define crew and main function
            write_crew_definition(file, crew_agents, task_vars)
            write_main_function(file)

        logging.info(f"\nYour CrewAI script is saved here: {script_file_path}")
    except IOError as e:
        logging.error(f"Error writing to file {script_file_path}: {e}")
        raise

def write_script_header(file):
    """Writes the header of the script including necessary imports."""
    file.write(
        'import os\n'
        'from crewai import Agent, Task, Crew, Process\n'
        'from langchain_openai import ChatOpenAI\n'
        'from langchain_community.tools import DuckDuckGoSearchRun\n'
        'from textwrap import dedent\n\n'
    )
def write_llm_configuration(file, llm_endpoint, llm_model, add_ollama_url, ollama_host, add_api_keys, openai_api_key, openai_model):
    """Writes the configuration for the LLM endpoint."""
    if llm_endpoint == 'openai':
        api_key_line = f'openai_api_key = "{openai_api_key}"\n' if add_api_keys else 'openai_api_key = os.getenv("OPENAI_API_KEY")\n'
        file.write(api_key_line)
        file.write(f'OpenAIGPT35 = ChatOpenAI(api_key=openai_api_key, model_name="{openai_model}", temperature=0.7)\n')
        file.write('llm = OpenAIGPT35\n\n')
    elif llm_endpoint == 'ollama':
        ollama_import_line = 'from langchain_community.llms import Ollama\n'
        ollama_config_line = f'OllamaInstance = Ollama(base_url="{ollama_host}", model="{llm_model}", verbose=True)\n' if add_ollama_url else ''
        file.write(ollama_import_line)
        file.write(ollama_config_line)
        file.write('llm = OllamaInstance\n\n')
def write_agents_and_tasks(file, agents_json):
    """Writes the agents and their tasks to the script."""
    task_vars, crew_agents = [], []
    # Extract a list of roles
    roles = [agent['role'] for agent in agents_json[0]]

    for agent in agents_json[0]:
        agent_var_name = agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_')
        crew_agents.append(f'agent_{agent_var_name}')
        
        file.write(f'agent_{agent_var_name} = Agent(\n')
        file.write(f'    role="{agent["role"]}",\n')
        file.write(f'    backstory=dedent("""{agent["backstory"]}"""),\n')
        file.write(f'    goal=dedent("""{agent["goal"]}"""),\n')
        file.write(f'    allow_delegation={agent["allow_delegation"]},\n')
        file.write(f'    verbose=True,\n')
        file.write(f'    llm=llm,\n')
        file.write(')\n')

        task_var_name = f'task_{agent_var_name}'
        task_vars.append(task_var_name)
        file.write(f'{task_var_name} = Task(\n')
        file.write(f'    description=dedent("""{agent["assigned_task"]}"""),\n')
        file.write(f'    agent=agent_{agent_var_name},\n')
        file.write(')\n\n')

    return task_vars, crew_agents
def write_crew_definition(file, crew_agents, task_vars):
    """Writes the crew definition to the script."""
    crew_definition = (
        'crew = Crew(\n'
        f'    agents=[{", ".join(crew_agents)}],\n'
        f'    tasks=[{", ".join(task_vars)}],\n'
        '    verbose=True,\n'
        '    process=Process.sequential,\n'
        ')\n\n'
    )
    file.write(crew_definition)
def write_main_function(file):
    """Writes the main function of the script."""
    main_function = (
        'if __name__ == "__main__":\n'
        '    print("## Welcome to Crew AI")\n'
        '    print("-------------------------------")\n'
        '    result = crew.kickoff()\n'
        '    print("\\n\\n########################")\n'
        '    print("## Here is your custom crew run result:")\n'
        '    print("########################\\n")\n'
        '    print(result)\n'
    )
    file.write(main_function)