import csv
import io
import os
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOpenAI
from crewai import Agent, Task, Crew, Process

# Initialize Ollama
def initialize_ollama(model='openhermes'):
    return Ollama(model=model, verbose=True)

# Get agent data from Ollama
def get_agent_data(ollama, goal):
    instruction = (
        f"Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with the goal: '{goal}'. "
        "Include columns 'role', 'goal', 'backstory', 'assigned_task', 'allow_delegation'. "
        "Each agent's details should be in quotes to avoid confusion with the comma delimiter. "
        "Provide a unique role, specific goal, brief backstory, assigned task, and delegation ability (True/False) for each agent."
    )
    return ollama.invoke(instruction)

# Parse CSV data from Ollama's response
def parse_csv_data(response):
    header = ['role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
    return [dict(zip(header, row + [None] * (len(header) - len(row))))
            for row in csv.reader(io.StringIO(response), delimiter=',')
            if row and 'role' not in row[0]]

# Define an agent for the CrewAI script
def define_agent(agent, search_tool):
    role_var = agent['role'].replace(" ", "_").replace("-", "_")
    delegation = 'True' if agent['allow_delegation'] == 'True' else 'False'
    return (f"{role_var} = Agent(\n"
            f"    role='{agent['role']}',\n"
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
            "from langchain_community.llms import Ollama\n"
            "from langchain_community.tools import DuckDuckGoSearchRun\n"
            "from langchain_community.chat_models import ChatOpenAI\n"
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
        goal = input("Please specify the overall goal: ")
        response = get_agent_data(ollama, goal)
        if not response:
            raise ValueError("No response from Ollama")

        agents_data = parse_csv_data(response)
        if not agents_data:
            raise ValueError("No agent data parsed")

        # Print the agent data
        print("Agent Data:")
        for agent in agents_data:
            print(f"\nAgent: {agent['role']}")
            print("-" * 50)
            print(f"- Goal: {agent['goal']}")
            print(f"- Backstory: {agent['backstory']}")
            print(f"- Assigned Task: {agent['assigned_task']}")
            print(f"- Allow Delegation: {agent['allow_delegation']}")
            print("-" * 50)

        file_path = os.path.join(os.getcwd(), 'crewai-script.py')
        write_crewai_script(agents_data, file_path, ollama, DuckDuckGoSearchRun())

        print(f"\nScript written to {file_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
