import csv
import io
import os
import traceback
from datetime import datetime
import argparse
import requests
from packaging import version
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process

# Autocrew version
autocrew_version = "1.0.4"

# Initialize Ollama
def initialize_ollama(model='openhermes'):
    return Ollama(model=model, verbose=True)

# Get agent data from Ollama
def get_agent_data(ollama, overall_goal, delimiter):
    print("Autocrew: Sending request to LLM...")
    instruction = (
        f'Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with the goal: "{overall_goal}". '
        f'Use the delimiter "{delimiter}" to separate the fields. '
        'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
        'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
        'Provide a single-word role, specific goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
    )
    response = ollama.invoke(instruction.format(overall_goal=overall_goal, delimiter=delimiter))
    print("\nOllama's CSV Output:")
    print(response)
    return response

# Save Ollama's CSV output to a file
def save_csv_output(response, overall_goal, index):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f'crewai-autocrew-{timestamp}-{overall_goal[:40].replace(" ", "-")}-{index}.csv'
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, 'w') as file:
        file.write(response)
    print(f'\nOllama\'s CSV output saved as {file_path}')

# Parse CSV data from Ollama's response
def parse_csv_data(response, delimiter=','):
    header = ['role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
    agents_data = []

    # Use the csv module to handle parsing
    csv_data = csv.reader(io.StringIO(response), delimiter=delimiter)
    lines = list(csv_data)

    header_line = lines[0]
    header_mapping = {h.lower(): h for h in header}
    header_indices = [header_mapping.get(h.lower()) for h in header_line]

    for line in lines[1:]:
        agent_data = {}
        for i, value in enumerate(line):
            header_name = header_indices[i]
            if header_name:
                agent_data[header_name] = value.strip('"')
        if 'role' not in agent_data or not agent_data['role']:
            raise ValueError('Role component missing in CSV data')
        agents_data.append(agent_data)
    return agents_data

# Define an agent for the CrewAI script
def define_agent(agent, search_tool):
    role_var = agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_')
    role_value = agent['role'].replace('"', '\\"')
    delegation = 'True' if agent['allow_delegation'] == 'True' else 'False'
    return (f'{role_var} = Agent(\n'
            f'    role="{role_value}",\n'
            f'    goal="{agent["goal"]}",\n'
            f'    backstory="{agent["backstory"]}",\n'
            '    verbose=True,\n'
            f'    allow_delegation={delegation},\n'
            '    llm=ollama_openhermes,\n'
            '    tools=[search_tool]\n'
            ')\n\n')

# Define a task for the CrewAI script
def define_task(agent):
    role_var = agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_')
    return (f'task_{role_var} = Task(\n'
            f'    description="{agent["assigned_task"].strip()}",\n'
            f'    agent={role_var},\n'
            '    verbose=True,\n'
            ')\n\n')

# Write the CrewAI script based on the agent and task data
def write_crewai_script(agents_data, crew_tasks, file_name, ollama_openhermes, search_tool):
    crew_agents = ', '.join([agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_') for agent in agents_data])

    with open(file_name, 'w') as file:
        # Writing imports and initializations
        file.write(
            'import os\n'
            'from langchain_community.chat_models import ChatOpenAI\n'
            'from langchain_community.llms import Ollama\n'
            'from langchain_community.tools import DuckDuckGoSearchRun\n'
            'from crewai import Agent, Task, Crew, Process\n\n'
            'os.environ["OPENAI_API_KEY"] = "your_OPENAI_api_key_here"\n\n'
            'ollama_openhermes = Ollama(model="openhermes")\n'
            'search_tool = DuckDuckGoSearchRun()\n\n'
        )

        for agent in agents_data:
            file.write(define_agent(agent, search_tool))
            file.write('\n')

        for agent in agents_data:
            file.write(define_task(agent))
            file.write('\n')

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

# Check the latest version of the script on GitHub
def check_latest_version():
    try:
        response = requests.get('https://raw.githubusercontent.com/yanniedog/crewai-autocrew/main/crewai-autocrew.py')
        response.raise_for_status()
        script_content = response.text
        version_line = next(line for line in script_content.split('\n') if line.startswith('autocrew_version = '))
        latest_version = version_line.split('=')[1].strip().strip('"')

        if version.parse(latest_version) > version.parse(autocrew_version):
            return latest_version
        else:
            return None

    except Exception as e:
        print(f'Error checking the latest version: {e}')
        return None

# Main function
def main():
    print()
    print(f"Autocrew (v{autocrew_version}) for CrewAI ")

    latest_version = check_latest_version()
    if latest_version and latest_version != autocrew_version:
        print(f'\n\033[1mNew version available: {latest_version}\033[0m')

    print("\nTo see the available command line parameters, type: python crewai-autocrew.py -h")
    print()
    parser = argparse.ArgumentParser(description='CrewAI Autocrew Script')
    parser.add_argument('overall_goal', nargs='?', type=str, help='The overall goal for the crew')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--autorun', action='store_true', help='Run the generated script automatically at the end')
    group.add_argument('-m', '--multiple', type=int, metavar='NUM_SCRIPTS', help='Create multiple CrewAI scripts for the same overall goal')
    args = parser.parse_args()

    if args.autorun and args.multiple:
        parser.error("The options -a/--autorun and -m/--multiple cannot be used together. Please choose one or the other.")

    overall_goal = args.overall_goal
    if not overall_goal:
        overall_goal = input('\033[1mPlease specify the overall goal:\033[0m ')

    num_scripts = args.multiple or 1

    try:
        ollama = initialize_ollama()
        delimiter = ','
        for i in range(num_scripts):
            response = get_agent_data(ollama, overall_goal, delimiter)
            if not response:
                raise ValueError('No response from Ollama')

            save_csv_output(response, overall_goal, i+1)

            agents_data = parse_csv_data(response, delimiter)
            if not agents_data:
                raise ValueError('No agent data parsed')

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            overall_goal_filename = overall_goal[:50].replace(' ', '-')
            file_name = f'crewai-autocrew-{timestamp}-{overall_goal_filename}-{i+1}.py'
            crewai_script_path = os.path.join(os.getcwd(), file_name)

            crew_tasks = ', '.join([f'task_{agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")}' for agent in agents_data])

            write_crewai_script(agents_data, crew_tasks, crewai_script_path, ollama, DuckDuckGoSearchRun())

            print(f'\nScript {i+1} written to {crewai_script_path}')

            if args.autorun:
                print('\nAutocrew: Running the generated CrewAI script...')
                os.system(f'python3 {crewai_script_path}')

    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

if __name__ == '__main__':
    main()
