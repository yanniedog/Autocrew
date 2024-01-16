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
    instruction = (
        f'Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with the goal: "{overall_goal}". '
        f'Use the delimiter "{delimiter}" to separate the fields. '
        'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
        'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
        'Provide a single-word role, specific goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
    )
    response = ollama.invoke(instruction.format(overall_goal=overall_goal, delimiter=delimiter))
    return response

# Save Ollama's CSV output to a file
def save_csv_output(response, overall_goal, index):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f'crewai-autocrew-{timestamp}-{overall_goal[:40].replace(" ", "-")}-{index}.csv'
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, 'w') as file:
        file.write(response)
    return file_path

# Parse CSV data from Ollama's response
def parse_csv_data(response, delimiter=',', filename=''):
    header = ['filename', 'role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
    agents_data = []

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
        agent_data['filename'] = filename  # Add the filename to the agent data
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
            f'    tools=[{search_tool}]\n'
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
def write_crewai_script(agents_data, crew_tasks, file_name):
    crew_agents = ', '.join([agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_') for agent in agents_data])

    with open(file_name, 'w') as file:
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
            file.write(define_agent(agent, "search_tool"))
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

# Rank the crews based on their likelihood of success
def rank_crews(csv_file_paths):
    ranked_crews = []
    overall_summary = ""

    ollama = initialize_ollama()  # Initialize Ollama once

    csv_file_paths = list(set(csv_file_paths))  # Remove duplicate file paths

    print('Invoking Ollama...')

    concatenated_csv_data = 'filename,role,goal,backstory,assigned_task,allow_delegation\n'  # Initialize the concatenated CSV data string

    for file_path in csv_file_paths:
        if "ranking" in file_path.lower():
            continue  # Skip processing if the filename contains "ranking"

        print(f'\nProcessing CSV: {file_path}')

        with open(file_path, 'r') as file:
            csv_data = file.read()

        filename = os.path.basename(file_path)  # Get the filename of the original CSV

        # Append the filename to each row in the CSV data
        csv_data_with_filename = '\n'.join([f'{filename},{row}' for row in csv_data.strip().split('\n')])

        concatenated_csv_data += csv_data_with_filename + '\n'  # Append the CSV data to the concatenated CSV

    print('\nConcatenated CSV Data:')
    print(concatenated_csv_data)

    ranked_crew = ollama.invoke(concatenated_csv_data)
    print('\nOllama Ranking:')
    print(ranked_crew)

    critique = ranked_crew  # Use the ranked_crew output as the critique
    print('\nOllama Critique:')
    print(critique)

    ranked_crews.append((csv_file_paths, ranked_crew, critique))
    overall_summary += f'\n\nCrews in the following CSV files:\n'
    for file_path in csv_file_paths:
        overall_summary += f'{file_path}\n'
    overall_summary += f'Ranking: {ranked_crew}\n'
    overall_summary += f'Critique: {critique}\n'

    overall_summary += f'\nOverall Summary:\n'
    overall_summary += f'Ollama has ranked the crews based on their likelihood of success.\n'
    overall_summary += f'It has provided a critique for each crew, highlighting their strengths and weaknesses.\n'
    overall_summary += f'The ranking and critique can be used to make informed decisions about the crews.\n'

    return ranked_crews, overall_summary

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
    parser.add_argument('-r', '--ranking', action='store_true', help='Perform ranking only based on existing CSV files')
    parser.add_argument('-m', '--multiple', action='store_true', help='Create multiple CrewAI scripts for the same overall goal')
    args = parser.parse_args()

    if args.ranking:
        overall_goal = args.overall_goal[:50].replace(' ', '-')
        csv_file_paths = [file for file in os.listdir() if file.startswith(f'crewai-autocrew-') and file.endswith('.csv') and overall_goal in file]
        if not csv_file_paths:
            print(f'No CSV files found for the provided overall goal: {args.overall_goal}')
            return

        try:
            ranked_crews, overall_summary = rank_crews(csv_file_paths)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            overall_goal_filename = overall_goal.replace('-', '_')
            ranked_crews_file_name = f'crewai-autocrew-{timestamp}-{overall_goal_filename}-ranking.csv'
            ranked_crews_file_path = os.path.join(os.getcwd(), ranked_crews_file_name)

            with open(ranked_crews_file_path, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['CSV File', 'Ranking', 'Critique'])
                for crew in ranked_crews:
                    writer.writerow([crew[0], crew[1], crew[2]])

            print(f'\nRanked crews saved as {ranked_crews_file_path}')
            print(f'\nOverall Summary:')
            print(overall_summary)

            # Provide the prompt to Ollama
            ollama = initialize_ollama()
            ollama.invoke(overall_summary)

        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

        return

    if args.overall_goal is None:
        overall_goal = input('\033[1mPlease specify the overall goal:\033[0m ')
    else:
        overall_goal = args.overall_goal

    num_scripts = 1
    if args.multiple:
        num_scripts = int(input('\033[1mPlease enter the number of different CrewAI scripts to create:\033[0m '))

    try:
        delimiter = ','
        csv_file_paths = []  # Initialize the list of CSV file paths
        for i in range(num_scripts):
            ollama = initialize_ollama()  # Initialize Ollama for each script
            response = get_agent_data(ollama, overall_goal, delimiter)
            if not response:
                raise ValueError('No response from Ollama')

            file_path = save_csv_output(response, overall_goal, i+1)

            agents_data = parse_csv_data(response, delimiter, filename=file_path)  # Pass the filename to the parse_csv_data function
            if not agents_data:
                raise ValueError('No agent data parsed')

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            overall_goal_filename = overall_goal[:50].replace(' ', '-')
            file_name = f'crewai-autocrew-{timestamp}-{overall_goal_filename}-{i+1}.py'
            crewai_script_path = os.path.join(os.getcwd(), file_name)

            crew_tasks = ', '.join([f'task_{agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")}' for agent in agents_data])

            write_crewai_script(agents_data, crew_tasks, crewai_script_path)

            print(f'\nScript {i+1} written to {crewai_script_path}')

            csv_file_paths.append(file_path)  # Add the CSV file path to the list

        if num_scripts > 1:
            ranked_crews, overall_summary = rank_crews(csv_file_paths)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            overall_goal_filename = overall_goal[:50].replace(' ', '-')
            ranked_crews_file_name = f'crewai-autocrew-{timestamp}-{overall_goal_filename}-ranking.csv'
            ranked_crews_file_path = os.path.join(os.getcwd(), ranked_crews_file_name)

            with open(ranked_crews_file_path, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['CSV File', 'Ranking', 'Critique'])
                for crew in ranked_crews:
                    writer.writerow([crew[0], crew[1], crew[2]])

            print(f'\nRanked crews saved as {ranked_crews_file_path}')
            print(f'\nOverall Summary:')
            print(overall_summary)

            # Provide the prompt to Ollama
            ollama = initialize_ollama()
            ollama.invoke(overall_summary)

    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

if __name__ == '__main__':
    main()
