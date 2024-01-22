import os
import subprocess
import logging

# Importing custom modules
from write_crewai_script import write_crewai_script
from check_latest_version import check_latest_version
from rank_crews import rank_crews

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main(crew_tasks, overall_goal, csv_file_paths, args):
    for i, file_path in enumerate(csv_file_paths):
        file_name = os.path.basename(file_path).replace('.csv', '.py')
        crewai_script_path = os.path.join(os.getcwd(), file_name)

        write_crewai_script(crew_tasks, crew_tasks, crewai_script_path, args.use_ollama_host)
        logger.info(f"Script {i + 1} written to {crewai_script_path}")

        if args.auto_run:
            logger.info(f'Automatically running script {i + 1}...')
            os.system(f'python3 {crewai_script_path}')

    if args.ranking:
        logger.info("Sending ranking request to Ollama...")
        if not csv_file_paths and args.overall_goal:
            csv_file_paths = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv') and args.overall_goal.replace(" ", "-") in f and any(greek_alpha in f for greek_alpha in greek_alphabets)]

        if csv_file_paths:
            ranked_crews, overall_summary = rank_crews(ollama, csv_file_paths, overall_goal, args.verbose)
            logger.info(overall_summary)

            import re
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
            if args.auto_run:
                logger.info(f'Automatically running the top-ranked script: {top_script_path}')
                os.system(f'python3 {top_script_path}')

if __name__ == "__main__":
    try:
        main(crew_tasks, overall_goal, csv_file_paths, args)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
