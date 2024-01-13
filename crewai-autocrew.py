import csv
import io
import os
from langchain.llms import Ollama
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process

# Initialize Ollama
def initialize_ollama(model='openhermes'):
    return Ollama(model=model, verbose=True)

# Get agent data from Ollama
def get_agent_data(ollama, overall_goal, delimiter):
    instruction = (
        f"Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with the goal: '{overall_goal}'. "
        f"Use the delimiter '{delimiter}' to separate the fields. "
        "Include columns 'role', 'goal', 'backstory', 'assigned_task', 'allow_delegation'. "
        "Each agent's details should be in quotes to avoid confusion with the delimiter. "
        "Provide a single-word role, specific goal, brief backstory, assigned task, and delegation ability (True/False) for each agent."
    )
    response = ollama.invoke(instruction.format(overall_goal=overall_goal, delimiter=delimiter))
    print("Ollama's CSV Output:")
    print(response)
    return response

# Parse CSV data from Ollama's response
def parse_csv_data(response, delimiter=','):
    header = ['role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
    agents_data = []
    lines = response.strip().split('\n')
    for line in lines[1:]:
        row = line.strip().split(delimiter)
        if len(row) > len(header):
            # Concatenate the role component if it is split across multiple fields
            role = delimiter.join(row[:len(header)-1])
            agent_data = dict(zip(header, [role] + row[len(header)-1:]))
        else:
            agent_data = dict(zip(header, row))
        if 'role' not in agent_data or not agent_data['role']:
            raise ValueError("Role component missing in CSV data")
        agents_data.append(agent_data)
    return agents_data

# Define an agent for the CrewAI script
def define_agent(agent, search_tool):
    role_var = agent['role'].replace(" ", "_").replace("-", "_")
    delegation = 'True' if agent['allow_delegation'] == 'True' else 'False'
    return (f"{role_var} = Agent(\n"
            f"    role=\"\"\"{agent['role']}\"\"\",\n"
            f"    goal=\"\"\"{agent['goal']}\"\"\",\n"
            f"    backstory=\"\"\"{agent['backstory']}\"\"\",\n"
            f"    verbose=True,\n"
            f"    allow_delegation={delegation},\n"
            f"    llm=ollama_openhermes,\n"
            f"    tools=[search_tool]\n"
            ")\n\n")

# Define a task for the CrewAI script
def define_task(agent):
    role_var = agent['role'].replace(" ", "_").replace("-", "_")
    return (f"task_{role_var} = Task(\n"
            f"    description=\"\"\"{agent['assigned_task']}\"\"\",\n"
            f"    agent={role_var},\n"
            f"    verbose=True,\n"
            ")\n\n")

# Write the CrewAI script based on the agent and task data
def write_crewai_script(agents_data, file_path, ollama_openhermes, search_tool):
    with open(file_path, 'w') as file:
        # Writing imports and initializations
        file.write(
            "import os\n"
            "from langchain_community.chat_models import ChatOpenAI\n"
            "from langchain_community.llms import Ollama\n"
            "from langchain_community.tools import DuckDuckGoSearchRun\n"
            "from crewai import Agent, Task, Crew, Process\n\n"
            "os.environ['OPENAI_API_KEY'] = 'your_OPENAI_api_key_here'\n\n"
            "ollama_openhermes = Ollama(model='openhermes')\n"
            "search_tool = DuckDuckGoSearchRun()\n\n"
        )

        for agent in agents_data:
            file.write(define_agent(agent, search_tool))

        for agent in agents_data:
            file.write(define_task(agent))

        crew_agents = ", ".join([agent['role'].replace(" ", "_").replace("-", "_") for agent in agents_data])
        crew_tasks = ", ".join([f"task_{agent['role'].replace(' ', '_').replace('-', '_')}" for agent in agents_data])

        file.write(
            "crew = Crew(\n"
            f"    agents=[{crew_agents}],\n"
            f"    tasks=[{crew_tasks}],\n"
            "    verbose=True,\n"
            "    process=Process.sequential,\n"
            ")\n\n"
            "# Kickoff the crew tasks\n"
            "result = crew.kickoff()\n\n"
            "# Handle the 'result' as needed\n"
        )

# Main function
def main():
    try:
        ollama = initialize_ollama()
        overall_goal = input("Please specify the overall goal: ")
        delimiter = input("Please specify the delimiter used in the CSV data: ")
        response = get_agent_data(ollama, overall_goal, delimiter)
        if not response:
            raise ValueError("No response from Ollama")

        agents_data = parse_csv_data(response, delimiter)
        if not agents_data:
            raise ValueError("No agent data parsed")

        file_path = os.path.join(os.getcwd(), 'crewai-script.py')
        write_crewai_script(agents_data, file_path, ollama, DuckDuckGoSearchRun())

        print(f"\nScript written to {file_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
