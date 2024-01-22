# Filename: get_next_crew_name.py

import os

def get_next_crew_name(overall_goal, greek_alphabets):
    # Replace spaces with hyphens in overall_goal for filename matching
    formatted_goal = overall_goal.replace(" ", "-")

    # Get all CSV files that include the overall_goal in their name
    existing_csv_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv') and formatted_goal in f]

    # Find the highest Greek alphabet index used in these filenames
    existing_indices = []
    for file_name in existing_csv_files:
        for greek_alpha in greek_alphabets:
            if greek_alpha in file_name:
                existing_indices.append(greek_alphabets.index(greek_alpha))
                break  # Stop after finding the first matching Greek alphabet

    # Determine the next Greek alphabet index
    next_index = max(existing_indices) + 1 if existing_indices else 0
    return greek_alphabets[next_index % len(greek_alphabets)]
