# Filename: define_task.py

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
