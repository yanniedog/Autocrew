# Filename: define_agent.py

def define_agent(agent, search_tool):
    role_var = agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_').replace(' ', '')
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
