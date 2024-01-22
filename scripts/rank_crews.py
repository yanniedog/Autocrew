# Filename: rank_crews.py

import csv
import io
import os

def rank_crews(ollama, csv_file_paths, overall_goal, verbose=False):
    ranked_crews = []
    overall_summary = ''
    concatenated_csv_data = 'crew_name,role,goal,backstory,assigned_task,allow_delegation\n'

    for file_path in csv_file_paths:
        try:
            with open(file_path, 'r') as file:
                csv_data = file.read().strip()

            if csv_data.count('\n') < 1:
                continue

            concatenated_csv_data += csv_data[csv_data.index('\n') + 1:] + '\n'

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if concatenated_csv_data.strip() == 'crew_name,role,goal,backstory,assigned_task,allow_delegation':
        print("Warning: No valid data found in the provided CSV files.")
        return [], "No ranking could be performed due to insufficient data."

    # Convert the concatenated CSV data to a JSON object
    json_data = []
    csv_reader = csv.DictReader(io.StringIO(concatenated_csv_data))
    for row in csv_reader:
        json_data.append(row)
    json_data_str = json.dumps(json_data)

    if verbose:
        print('\nConcatenated CSV Data:')
        print(concatenated_csv_data)

    crew_names_str = ', '.join([os.path.basename(file_path).split('-')[-1].split('.')[0] for file_path in csv_file_paths])

    # Construct and print the Ollama prompt with the crew names
    prompt = (
        f"Analyze the following list of crews ({crew_names_str}) to determine their suitability for successfully completing the task: "
        f"{overall_goal}. The crews are represented in a JSON object format: {json_data_str}. "
        "Please provide a ranking of the crews by their names, with the most suitable crew listed first. "
        "Also, provide a brief critique for each crew, highlighting their strengths and weaknesses."
    )

    if verbose:
        print("Prompt to be sent to Ollama:\n", prompt)

    # Invoke Ollama with the prompt and JSON object
    ranked_crew = ollama.invoke(prompt)

    ranked_crews.append((concatenated_csv_data, ranked_crew))
    overall_summary += f'\n\nCrews in the following CSV files:\n'
    for file_path in csv_file_paths:
        overall_summary += f'{file_path}\n'
    overall_summary += f'Ranking: {ranked_crew}\n'

    overall_summary += f'\nOverall Summary:\n'
    overall_summary += f'Ollama has ranked the crews based on their likelihood of success.\n'
    overall_summary += f'It has provided a critique for each crew, highlighting their strengths and weaknesses.\n'
    overall_summary += f'The ranking and critique can be used to make informed decisions about the crews.\n'

    return ranked_crews, overall_summary
