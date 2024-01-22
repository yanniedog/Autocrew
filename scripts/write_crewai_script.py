# Filename: write_crewai_script.py

import os

def write_crewai_script(agents_data, crew_tasks, file_name, use_ollama_host):
    crew_agents = ', '.join([agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_') for agent in agents_data])

    with open(file_name, 'w') as file:
        file.write(
            'import os\n'
            'from langchain_community.chat_models import ChatOpenAI\n'
            'from langchain_community.llms import Ollama\n'
            'from langchain_community.tools import DuckDuckGoSearchRun\n'
            'from crewai import Agent, Task, Crew, Process\n\n'
            'os.environ["OPENAI_API_KEY"] = "your_OPENAI_api_key_here"\n\n'
        )

        if use_ollama_host:
            file.write(f'ollama_host = "{ollama_host}"\n')  # Write the ollama_host variable to the generated script
            file.write('ollama_openhermes = Ollama(model="openhermes", base_url=ollama_host)\n')  # Use ollama_host to initialize Ollama
        else:
            file.write('ollama_openhermes = Ollama(model="openhermes")\n')

        file.write(
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
