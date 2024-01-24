# script_generation.py
import os
import subprocess
import logging
import argparse
import re
import traceback

# Importing custom modules
from write_crewai_script import write_crewai_script  
from check_latest_version import check_latest_version
from rank_crews import rank_crews

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_task_var_name(role):
    return f'task_{role.replace(" ", "_").replace("-", "_").replace(".", "_")}'

def generate_crew_tasks(agents_data):
    return ', '.join([f'task_{agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")}' for agent in agents_data])

def main(crew_tasks, overall_goal, csv_file_paths, args, ollama):
    for i, file_path in enumerate(csv_file_paths):
        file_name = os.path.basename(file_path).replace('.csv', '.py')
        crewai_script_path = os.path.join(os.getcwd(), file_name)
        
        # Corrected the arguments for write_crewai_script
        write_crewai_script(crew_tasks, crew_tasks, crewai_script_path, args.use_ollama_host)
        logger.info(f"Script {i + 1} written to {crewai_script_path}")
        
        if args.auto_run:
            logger.info(f'Automatically running script {i + 1}...')
            # Using subprocess.run instead of os.system
            subprocess.run(['python3', crewai_script_path], check=True)
            
    if args.ranking:
        logger.info("Sending ranking request to Ollama...")
        greek_alphabets = ['alpha', 'beta', 'gamma', 'delta']  # Define or import this list as required
        if not csv_file_paths and args.overall_goal:
            csv_file_paths = [f for f in os.listdir(os.getcwd()) 
                if f.endswith('.csv') and args.overall_goal.replace(" ", "-") in f  
                and any(greek_alpha in f for greek_alpha in greek_alphabets)]

        if csv_file_paths:
            ranked_crews, overall_summary = rank_crews(ollama, csv_file_paths, overall_goal, args.verbose)
            logger.info(overall_summary)

            top_crew_name_search = re.search(r'"(.+?)"', overall_summary)
            if top_crew_name_search:
                top_crew_name = top_crew_name_search.group(1)
            else:
                logger.error("Top-ranked crew name not found in the overall summary.")

            if args.auto_run:
                overall_goal_formatted = overall_goal.replace(" ", "-")
                script_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.py')] 

                for script_file in script_files:
                    if overall_goal_formatted in script_file and top_crew_name in script_file:
                        top_script_path = os.path.join(os.getcwd(), script_file)
                        break

            if args.verbose:
                logger.info(f"Top-ranked crew: {top_crew_name}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Script generation script")    
    parser.add_argument("--use_ollama_host", help="Use Ollama host", type=bool, default=False)
    parser.add_argument("--auto_run", help="Automatically run scripts", action="store_true")
    parser.add_argument("--ranking", help="Rank the crews", action="store_true")
    parser.add_argument("--overall_goal", help="Overall goal of the crews", type=str)
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    args = parser.parse_args()
    
    # Define these variables or pass them in some other way
    crew_tasks = []  # Should be defined or passed in appropriately
    overall_goal = ""  # Should be defined or passed in appropriately
    csv_file_paths = []  # Should be defined or passed in appropriately
    ollama = None  # Should be defined or passed in appropriately

    try:
        main(crew_tasks, overall_goal, csv_file_paths, args, ollama)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
