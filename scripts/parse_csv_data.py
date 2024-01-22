# Filename: parse_csv_data.py

import csv
import io

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
                if header_name == 'assigned_task':
                    # Replace commas with 'and' within square brackets and remove the square brackets
                    value = value.replace(',', ' and ').replace('[', '').replace(']', '')
                    agent_data[header_name] = value.strip()
                else:
                    agent_data[header_name] = value.strip('"')
        if 'role' not in agent_data or not agent_data['role']:
            raise ValueError('Role component missing in CSV data')
        agent_data['filename'] = filename  # Add the filename to the agent data
        agents_data.append(agent_data)

    return agents_data
