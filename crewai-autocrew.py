import argparse
import csv
import os
import io
import sys
import traceback
import logging
from datetime import datetime
import requests
from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from packaging import version
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Autocrew version
autocrew_version = "1.2.1"


def initialize_ollama(model='openhermes'):
    return Ollama(model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


def add_filename_to_csv(file_path, crew_name):
    modified_rows = []
    filename = os.path.basename(file_path)

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header row
        modified_header = header + ['file_name', 'crew_name']
        modified_rows.append(modified_header)

        for row in csv_reader:
            modified_row = row + [f'"{filename}"', f'"{crew_name}"']
            modified_rows.append(modified_row)

    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(modified_rows)


def get_agent_data(ollama, overall_goal, delimiter):
    instruction = (
        f'Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with the goal: "{overall_goal}". '
        f'Use the delimiter "{delimiter}" to separate the fields. '
        'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
        'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
        'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
    )
    response = ollama.invoke(instruction.format(overall_goal=overall_goal, delimiter=delimiter))
    return response

def get_next_crew_name(overall_goal, greek_alphabets):
    # Replace spaces with hyphens in overall_goal for filename matching
    formatted_goal = overall_goal.replace(" ", "-")

    # Get all CSV files that include the overall_goal in their name
    existing_csv_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv') and formatted_goal in f]

    # Find the highest Greek alphabet index used in these filenames
    existing_indices = []
    for file_name in existing_csv_files:
        for greek_alpha in greek_alphabets:
            if greek_alpha in file_name:
                existing_indices.append(greek_alphabets.index(greek_alpha))
                break  # Stop after finding the first matching Greek alphabet

    # Determine the next Greek alphabet index
    next_index = max(existing_indices) + 1 if existing_indices else 0
    return greek_alphabets[next_index % len(greek_alphabets)]


def save_csv_output(response, overall_goal, greek_alphabets):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    crew_name = get_next_crew_name(overall_goal, greek_alphabets)
    clean_crew_name = crew_name.strip('"')  # Remove quotes for file name
    file_name = f'crewai-autocrew-{timestamp}-{overall_goal[:40].replace(" ", "-")}-{clean_crew_name}.csv'
    file_path = os.path.join(os.getcwd(), file_name)

    # Split the response into lines
    lines = response.split('\n')

    # Write the modified response to the file
    with open(file_path, 'w') as file:
        # Write the header row
        file.write("file_name,crew_name," + lines[0] + '\n')

        # Modify and write the data rows
        for line in lines[1:]:
            if line.strip():
                modified_line = f'{file_name},{crew_name},{line}\n'
                file.write(modified_line)

    return file_path



def parse_csv_data(response, delimiter=',', filename=''):
    header = ['role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
    agents_data = []

    csv_data = csv.reader(io.StringIO(response), delimiter=delimiter)
    lines = list(csv_data)

    header_line = lines[0]
    header_mapping = {h.lower(): h for h in header}
    header_indices = [header_mapping.get(h.lower()) for h in header_line]

    if not header_indices:
        raise ValueError('Header component missing in CSV data')

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


def define_agent(agent, search_tool):
    role_var = agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_')
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
        f'    llm=ollama_openhermes,\n'
        f'    tools=[{search_tool}]\n'
        ')\n\n'
    )


def get_task_var_name(role):
    return f'task_{role.replace(" ", "_").replace("-", "_").replace(".", "_")}'


def define_task(agent):
    task_var = get_task_var_name(agent['role'])

    # Escape double quotes in assigned_task if needed
    task_description = agent["assigned_task"].strip().replace('"', '\\"')

    return (
        f'{task_var} = Task(\n'
        f' description="{task_description}",\n'
        f' agent={agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")},\n'
        ' verbose=True,\n'
        ')\n\n'
    )


def generate_crew_tasks(agents_data):
    return ', '.join([f'task_{agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")}' for agent in agents_data])


def write_crewai_script(agents_data, crew_tasks, file_name):
    crew_agents = ', '.join([agent['role'].replace(' ', '').replace('-', '').replace('.', '_') for agent in agents_data])

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


def ranked_crews(ollama, csv_file_paths, overall_goal, verbose=False):
    concatenated_csv_data = 'crew_name,role,goal,backstory,assigned_task,allow_delegation\n'

    for file_path in csv_file_paths:
        try:
            with open(file_path, 'r') as file:
                csv_data = file.read().strip()

            # Skip if only header is present in the file
            if csv_data.count('\n') < 1:
                continue

            # Skip the header of each file and concatenate the rest
            csv_data_without_header = csv_data[csv_data.index('\n') + 1:]
            concatenated_csv_data += csv_data_without_header + '\n'
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Check if concatenated data is empty (other than the header)
    if concatenated_csv_data.strip() == 'crew_name,role,goal,backstory,assigned_task,allow_delegation':
        error_message = "Error: Concatenated CSV data is empty. No valid data found in the provided CSV files."
        debug_info = f"Checked CSV files: {csv_file_paths}"
        raise ValueError(f"{error_message}\nDebug Information: {debug_info}")

    # Save the concatenated CSV data to a file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    overall_goal_formatted = overall_goal.replace(" ", "-")
    concatenated_file_name = f'crewai-autocrew-{timestamp}-{overall_goal_formatted}-concatenated.csv'
    concatenated_file_path = os.path.join(os.getcwd(), concatenated_file_name)

    with open(concatenated_file_path, 'w') as file:
        file.write(concatenated_csv_data)

    # Print concatenated CSV data only if verbose flag is set
    if verbose:
        print('\nConcatenated CSV Data:')
        print(concatenated_csv_data)

    # Construct and print the Ollama prompt if verbose
    prompt = (
        "Analyze the following list of crews to determine their suitability for successfully completing the task: "
        f"{overall_goal}. [rest of the prompt]..."
    )
    if verbose:
        print("Prompt to be sent to Ollama:\n", prompt)

    # Invoke Ollama with the prompt and ensure output is streamed
    ranked_crew = ollama.invoke(prompt)

    ranked_crews.append((concatenated_file_path, ranked_crew))
    overall_summary += f'\n\nCrews in the following CSV files:\n'
    for file_path in csv_file_paths:
        overall_summary += f'{file_path}\n'
    overall_summary += f'Ranking: {ranked_crew}\n'

    overall_summary += f'\nOverall Summary:\n'
    overall_summary += f'Ollama has ranked the crews based on their likelihood of success.\n'
    overall_summary += f'It has provided a critique for each crew, highlighting their strengths and weaknesses.\n'
    overall_summary += f'The ranking and critique can be used to make informed decisions about the crews.\n'

    return ranked_crews, overall_summary


def main():
    rank_crews = []
    ranked_crews = []
    greek_alphabets = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
                       "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]
    print(f"\nAutocrew (v{autocrew_version}) for CrewAI\n")

    latest_version = check_latest_version()
    if latest_version and latest_version != autocrew_version:
        print(f'\nNew version available: {latest_version}\n')

    print("\nTo see the available command line parameters, type: python3 crewai-autocrew.py -h\n")

    parser = argparse.ArgumentParser(description='CrewAI Autocrew Script')
    parser.add_argument('overall_goal', nargs='?', type=str, help='The overall goal for the crew')
    parser.add_argument('-a', '--auto_run', action='store_true', help='Automatically run the generated script')
    parser.add_argument('-m', '--multiple', type=int, metavar='NUM', help='Create NUM number of CrewAI scripts for the same overall goal. Example: -m 3')
    parser.add_argument('-r', '--ranking', action='store_true', help='Perform ranking only based on existing CSV files --> currently EXPERIMENTAL')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    
    # Check if the script is in ranking mode and an overall goal is provided
    if args.ranking and args.overall_goal:
        # Search for CSV files in the current working directory
        csv_file_paths = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv') and args.overall_goal in f]

        if not csv_file_paths:
            print(f"No CSV files found for the overall goal: {args.overall_goal}")
        else:
            print(f"Found the following CSV files for the overall goal: {args.overall_goal}")
            for csv_file_path in csv_file_paths:
                print(csv_file_path)
    else:
        # Initial summary of actions based on arguments
        print("\nInitial Summary of Actions:")
        if args.overall_goal:
            print(f"  - Overall goal specified: {args.overall_goal}")
        if args.auto_run:
            print("  - The script(s) will be automatically run after generation.")
        if args.multiple:
            print(f"  - Number of scripts to generate: {args.multiple}")
        if args.ranking:
            print("  - Ranking mode activated. Existing CSV files will be used for ranking.")
        if args.verbose:
            print("  - Verbose mode activated. Additional details will be provided during execution.")
        print()

        if args.ranking and not args.overall_goal:
            print("Warning: Ranking mode requires an overall goal. Please provide an overall goal using the command line or by entering it when prompted.")
            overall_goal = input('\nPlease specify the overall goal: \n')
        else:
            overall_goal = args.overall_goal

        csv_file_paths = []  # Initialize the list of CSV file paths

        ollama = initialize_ollama()

        # Perform ranking using the -r option
        if args.ranking:
            print("Ranking mode activated...")

            # Ensure an overall goal is provided
            if not overall_goal:
                print("Error: Overall goal is required for ranking.")
                return

            print(f"Overall goal for ranking: {overall_goal}")

            # Identify relevant CSV files
            csv_file_paths = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv') and overall_goal in f]
            if not csv_file_paths:
                print(f"No CSV files found for the overall goal: {overall_goal}")
                return

            print(f"CSV files identified for ranking: {csv_file_paths}")

            # Proceed with ranking
            ranked_crews, overall_summary = rank_crews(ollama, csv_file_paths, overall_goal, args.verbose)
            print(overall_summary)
            if args.auto_run:
                # Extract the top-ranked crew name
                top_crew_line = overall_summary.split('\n')[0]
                top_crew_name = top_crew_line.split(':')[1].strip().strip('"')

                # Find the script file path corresponding to the top-ranked crew
                top_script_path = None
                for file_path in csv_file_paths:
                    if top_crew_name in file_path:
                        top_script_path = file_path.replace('.csv', '.py')
                        break

                # Execute the top-ranked script
                if top_script_path:
                    print(f'\nAutomatically running the top-ranked script: {top_script_path}\n')
                    os.system(f'python3 {top_script_path}')
                else:
                    print("\nTop-ranked script not found. Please ensure the script files are in the correct directory.\n")

if __name__ == '__main__':
    main()