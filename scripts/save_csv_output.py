# Filename: save_csv_output.py

from datetime import datetime
import os

def save_csv_output(response, overall_goal, greek_alphabets):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    crew_name = get_next_crew_name(overall_goal, greek_alphabets)
    clean_crew_name = crew_name.strip('"')  # Remove quotes for file name
    file_name = f'crewai-autocrew-{timestamp}-{overall_goal[:40].replace(" ", "-")}-{clean_crew_name}.csv'
    file_path = os.path.join(os.getcwd(), file_name)

    # Split the response into lines
    lines = response.split('\n')

    # Write the modified response to the file
    with open(file_path, 'w') as file:
        # Write the header row
        file.write("crew_name," + lines[0] + '\n')

        # Modify and write the data rows
        for line in lines[1:]:
            if line.strip():
                # Add crew_name to each line to include it in the concatenated CSV
                modified_line = f'"{crew_name}",{line}\n'
                file.write(modified_line)

    return file_path
