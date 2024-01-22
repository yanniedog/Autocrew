# Filename: generate_crew_tasks.py

def generate_crew_tasks(agents_data):
    return ', '.join([f'task_{agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")}' for agent in agents_data])
