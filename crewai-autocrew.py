import csv
import io
import os
import traceback
from datetime import datetime
import argparse
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process

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
def save_csv_output(response, overall_goal):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f'crewai-autocrew-{timestamp}-{overall_goal.replace(" ", "-")}.csv'
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

# Main function
def main():
    parser = argparse.ArgumentParser(description='CrewAI Autocrew Script')
    parser.add_argument('overall_goal', nargs='?', type=str, help='The overall goal for the crew')
    parser.add_argument('-a', '--autorun', action='store_true', help='Run the generated script automatically at the end')
    args = parser.parse_args()

    overall_goal = args.overall_goal
    if not overall_goal:
        overall_goal = input('Please specify the overall goal: ')

    try:
        ollama = initialize_ollama()
        delimiter = ','
        response = get_agent_data(ollama, overall_goal, delimiter)
        if not response:
            raise ValueError('No response from Ollama')

        save_csv_output(response, overall_goal)

        agents_data = parse_csv_data(response, delimiter)
        if not agents_data:
            raise ValueError('No agent data parsed')

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        overall_goal_filename = overall_goal.replace(' ', '-')
        file_name = f'crewai-autocrew-{timestamp}-{overall_goal_filename}.py'
        crewai_script_path = os.path.join(os.getcwd(), file_name)

        crew_tasks = ', '.join([f'task_{agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")}' for agent in agents_data])

        write_crewai_script(agents_data, crew_tasks, crewai_script_path, ollama, DuckDuckGoSearchRun())

        print(f'\nScript written to {crewai_script_path}')

        if args.autorun:
            print('\nAutocrew: Running the generated CrewAI script...')
            os.system(f'python3 {crewai_script_path}')

    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

if __name__ == '__main__':
    main()
