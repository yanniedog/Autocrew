# get_agent_data.py

instruction = (
    f'Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with a goal. '
    f'Use a specified delimiter to separate the fields. '
    'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
    'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
    'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
)

def invoke_ollama(ollama, overall_goal, delimiter):
    response = ollama.invoke(instruction.format(overall_goal=overall_goal, delimiter=delimiter))
    return response
