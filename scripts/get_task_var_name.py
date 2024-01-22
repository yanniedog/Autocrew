# Filename: get_task_var_name.py

def get_task_var_name(role):
    return f'task_{role.replace(" ", "_").replace("-", "_").replace(".", "_")}'
