# utils.py

import csv
import io
import logging
import os
import re
import time

from textwrap import dedent
from datetime import datetime

# Assuming tiktoken is a custom or third-party library you have access to
import tiktoken

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
    formatted_goal = overall_goal.replace(" ", "-")
    existing_files = [f for f in os.listdir(script_directory) if (f.endswith('.csv') or f.endswith('.py')) and formatted_goal in f]
    existing_indices = [GREEK_ALPHABETS.index(f.split('-')[-1].split('.')[0]) for f in existing_files if f.split('-')[-1].split('.')[0] in GREEK_ALPHABETS]
    next_index = (max(existing_indices) + 1) % len(GREEK_ALPHABETS) if existing_indices else 0
    return GREEK_ALPHABETS[next_index]

def parse_csv_data(response, delimiter=',', filename=''):
    """Parses CSV data from a string response."""
    csv_pattern = r'("role","goal","backstory","assigned_task","allow_delegation".*?)(?:```|$)'
    match = re.search(csv_pattern, response, re.DOTALL)
    if not match:
        logging.error("CSV data not found in the response.")
        raise ValueError('CSV data not found in the response')

    csv_data = match.group(1).strip()  # Remove any extra whitespace
    logging.debug(f"Extracted CSV data for parsing:\n{csv_data}")

    header = ['role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
    agents_data = []

    try:
        csv_reader = csv.reader(io.StringIO(csv_data), delimiter=delimiter)
        lines = list(csv_reader)
    except Exception as e:
        logging.error(f"Error reading CSV data: {e}")
        raise

    if not lines:
        logging.error("CSV data is empty after splitting into lines.")
        raise ValueError('CSV data is empty')

    header_line = lines[0]
    header_indices = {h.lower(): i for i, h in enumerate(header_line)}

    for required_header in header:
        if required_header not in header_indices:
            logging.error(f'Missing required header "{required_header}" in CSV data')
            raise ValueError(f'Missing required header "{required_header}"')

    for line in lines[1:]:  # Skip the header line
        agent_data = {}
        for header_name in header:
            header_index = header_indices.get(header_name.lower())
            agent_data[header_name] = line[header_index].strip('"').strip() if header_index is not None and header_index < len(line) else None
        if 'role' not in agent_data or not agent_data['role']:
            logging.error('Role component missing in line of CSV data')
            raise ValueError('Role component missing in CSV data')
        agent_data['filename'] = filename
        agents_data.append(agent_data)

    logging.debug(f"Successfully parsed {len(agents_data)} agents from CSV data.")
    return agents_data

def save_csv_output(response, overall_goal, script_directory="scripts", truncation_length=40, greek_suffix=None):
    """Saves the CSV output to a file."""
    reader = csv.reader(io.StringIO(response), quotechar='"', delimiter=',', skipinitialspace=True)
    
    cleaned_csv_lines = []
    for fields in reader:
        if len(fields) != 5:
            continue
        cleaned_fields = ['"{}"'.format(field.replace('"', '""')) for field in fields]
        cleaned_line = ','.join(cleaned_fields)
        cleaned_csv_lines.append(cleaned_line)

    if cleaned_csv_lines:
        csv_data = '\n'.join(cleaned_csv_lines)
        logging.debug("Extracted and cleaned CSV data from raw output.")
        logging.info(f"\nDetails of your auto-generated AI crew:\n\n{csv_data}")
    else:
        logging.error("No CSV data found in the response.")
        raise ValueError("No CSV data found in the response")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use the provided greek_suffix if available, otherwise get the next one
    greek_suffix = greek_suffix or get_next_crew_name(overall_goal, script_directory)
    # Truncate the overall_goal to the specified number of characters for the filename
    truncated_goal = overall_goal[:truncation_length].replace(" ", "-")
    file_name = f'crewai-autocrew-{timestamp}-{truncated_goal}-{greek_suffix}.csv'
    directory = os.path.join(os.getcwd(), script_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    
    with open(file_path, 'w') as file:
        file.write(f'# {file_path}\n')
        file.write(csv_data)
        logging.debug(f"CSV file saved at: {file_path}")

    return file_path



def countdown_timer(seconds: int):
    """Displays a countdown timer for the specified number of seconds."""
    for i in range(seconds, 0, -1):
        logging.debug(f"Pausing for {i} seconds")
        time.sleep(1)
    logging.info("Continuing...")

def redact_api_key(api_key):
    """Redacts all but the last 4 characters of the API key."""
    return '*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else api_key

def write_crewai_script(agents_data, crew_tasks, file_name, llm_endpoint_within_generated_scripts, llm_model_within_generated_scripts, add_ollama_host_url_to_crewai_scripts, ollama_host, add_api_keys_to_crewai_scripts, openai_api_key, openai_model):
    # Define the path for the script file
    script_file_path = os.path.join("scripts", file_name)

    # Open the file and start writing the script content
    with open(script_file_path, 'w') as file:
        # Script header and imports
        file.write(
            'import os\n'
            'from crewai import Agent, Task, Crew, Process\n'
            'from langchain_openai import ChatOpenAI\n'
            'from langchain_community.tools import DuckDuckGoSearchRun\n'
            'from textwrap import dedent\n\n'
        )

        # Check and write LLM configuration based on settings
        if llm_endpoint_within_generated_scripts == 'openai':
            if add_api_keys_to_crewai_scripts:
                file.write(f'openai_api_key = "{openai_api_key}"\n')
            else:
                file.write('openai_api_key = os.getenv("OPENAI_API_KEY")\n')
            file.write(f'OpenAIGPT35 = ChatOpenAI(api_key=openai_api_key, model_name="{openai_model}", temperature=0.7)\n')
            llm_var = 'OpenAIGPT35'
        elif llm_endpoint_within_generated_scripts == 'ollama':
            # Import the Ollama class and instantiate it
            file.write('from langchain_community.llms import Ollama\n')
            file.write(f'OllamaInstance = Ollama(base_url="{ollama_host}", model="{llm_model_within_generated_scripts}", verbose=True)\n')
            llm_var = 'OllamaInstance'
        # You can add more conditions here if there are other LLM endpoints to handle

        # Ensure llm_var is defined
        if llm_var is None:
            raise ValueError("LLM variable not set. Check your configuration.")

        # Define agents and their tasks
        task_vars = []  # List to keep track of task variable names
        crew_agents = []  # List to keep track of agent variable names

        for agent in agents_data:
            agent_var_name = agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_')
            crew_agents.append(f'agent_{agent_var_name}')  # Add the agent variable name to the list
            file.write(f'agent_{agent_var_name} = Agent(\n')
            file.write(f'    role="{agent["role"]}",\n')
            file.write(f'    backstory=dedent("""{agent["backstory"]}"""),\n')
            file.write(f'    goal=dedent("""{agent["goal"]}"""),\n')
            file.write(f'    allow_delegation={agent["allow_delegation"]},\n')
            file.write(f'    verbose=True,\n')
            file.write(f'    llm={llm_var},\n')
            file.write(')\n')
            # Update Task instantiation to use keyword arguments
            task_var_name = f'task_{agent_var_name}'
            task_vars.append(task_var_name)
            file.write(f'{task_var_name} = Task(\n')
            file.write(f'    description=dedent("""{agent["assigned_task"]}"""),\n')
            file.write(f'    agent=agent_{agent_var_name},\n')
            file.write(')\n')

        # Define crew
        file.write(
            'crew = Crew(\n'
            f'    agents=[{", ".join(crew_agents)}],\n'  # Use the list of agent variable names
            f'    tasks=[{", ".join(task_vars)}],\n'
            '    verbose=True,\n'
            '    process=Process.sequential,\n'
            ')\n\n'
            'result = crew.kickoff()\n\n'
        )

        # Main function
        file.write(
            'if __name__ == "__main__":\n'
            '    print("## Welcome to Crew AI")\n'
            '    print("-------------------------------")\n'
            '    result = crew.kickoff()\n'
            '    print("\\n\\n########################")\n'
            '    print("## Here is your custom crew run result:")\n'
            '    print("########################\\n")\n'
            '    print(result)\n'
        )

    logging.info(f"\nYour CrewAI script is saved here:\n {script_file_path}\n")  # Log the full path of the saved file
