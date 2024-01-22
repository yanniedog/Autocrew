# Filename: autocrew.py

import os
import sys
from pathlib import Path
import configparser

# Add the scripts directory to the system path
scripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.append(scripts_path)

# Import the necessary functions from the smaller scripts
from get_ngrok_public_url import get_ngrok_public_url
from initialize_ollama import initialize_ollama
from get_agent_data import get_agent_data
from get_next_crew_name import get_next_crew_name
from save_csv_output import save_csv_output
from parse_csv_data import parse_csv_data
from define_agent import define_agent
from get_task_var_name import get_task_var_name
from define_task import define_task
from generate_crew_tasks import generate_crew_tasks
from write_crewai_script import write_crewai_script
from check_latest_version import check_latest_version
from rank_crews import rank_crews

# Read the config file
config = configparser.ConfigParser()
config.read(os.path.join(Path.home(), "autocrew", "config.ini"))

# Call the functions in the correct order
ngrok_url = get_ngrok_public_url()
ollama = initialize_ollama(use_ollama_host=True)
agent_data = get_agent_data(ollama, "Sample Goal", delimiter=',')
next_crew_name = get_next_crew_name("Sample Goal", ["alpha", "beta", "gamma"])
csv_output_path = save_csv_output(agent_data, "Sample Goal", ["alpha", "beta", "gamma"])
agents_data = parse_csv_data(agent_data, delimiter=',', filename=csv_output_path)
agent_definition = define_agent(agents_data[0], "search_tool")
task_var_name = get_task_var_name("Sample Role")
task_definition = define_task(agents_data[0])
crew_tasks = generate_crew_tasks(agents_data)
write_crewai_script(agents_data, crew_tasks, "sample_output.py", use_ollama_host=True)
latest_version = check_latest_version()
ranked_crews, overall_summary = rank_crews(ollama, [csv_output_path], "Sample Goal", verbose=True)

# Print the results
print(f"ngrok URL: {ngrok_url}")
print(f"Generated script path: {csv_output_path}")
print(f"Latest version: {latest_version}")
print(f"Overall summary: {overall_summary}")
