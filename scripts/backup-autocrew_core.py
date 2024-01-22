import argparse
import csv
import io
import json
import os
import requests
import traceback
from datetime import datetime
from packaging import version
from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging
from typing import Any, Dict, List
import subprocess

# Importing custom modules
from get_ngrok_public_url import get_ngrok_public_url
from initialize_ollama import initialize_ollama
from get_agent_data import get_agent_data
from get_next_crew_name import get_next_crew_name
from save_csv_output import save_csv_output
from parse_csv_data import parse_csv_data
from define_agent import define_agent
from get_task_var_name import get_task_var_name
from define_task import define_task
from generate_crew_tasks import generate_crew_tasks
from write_crewai_script import write_crewai_script
from check_latest_version import check_latest_version
from rank_crews import rank_crews

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

autocrew_version = "1.3"

def main(config):
    greek_alphabets = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
                       "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]
    logger.info(f"Autocrew (v{autocrew_version}) for CrewAI")

    ngrok_url = get_ngrok_public_url()
    if ngrok_url:
        logger.info(f"ngrok URL: {ngrok_url}")

    ollama_host = ""
    if os.getenv('OLLAMA_HOST'):
        ollama_host = os.getenv('OLLAMA_HOST')
        logger.info(f"OLLAMA_HOST: {ollama_host}")

    logger.info("To see the available command line parameters, type: python3 autocrew.py --help")

    parser = argparse.ArgumentParser(description='CrewAI Autocrew Script')
    parser.add_argument('overall_goal', nargs='?', type=str, help='The overall goal for the crew')
    parser.add_argument('-a', '--auto_run', action='store_true', help='Automatically run the generated script')
    parser.add_argument('-m', '--multiple', type=int, metavar='NUM', help='Create NUM number of CrewAI scripts for the same overall goal. Example: -m 3')
    parser.add_argument('-r', '--ranking', action='store_true', help='Perform ranking only based on existing CSV files --> currently EXPERIMENTAL')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--use_ollama_host', action='store_true', help='Use OLLAMA_HOST from the original script in the generated script')

    args = parser.parse_args()

    logger.info("Initial Summary of Actions:")
    if args.overall_goal:
        logger.info(f"Overall goal specified: {args.overall_goal}")
    if args.auto_run:
        logger.info("The script(s) will be automatically run after generation.")
    if args.multiple:
        logger.info(f"Number of scripts to generate: {args.multiple}")
    if args.ranking:
        logger.info("Ranking mode activated. Existing CSV files will be used for ranking.")
    if args.verbose:
        logger.info("Verbose mode activated. Additional details will be provided during execution.")
    if args.use_ollama_host:
        logger.info("Use OLLAMA_HOST from the original script in the generated script.")

    if args.ranking and not args.overall_goal:
        logger.warning("Ranking mode requires an overall goal. Please provide an overall goal using the command line or by entering it when prompted.")
    
    if args.overall_goal is None:
        overall_goal = input('\nPlease specify the overall goal: \n')
    else:
        overall_goal = args.overall_goal

    if args.multiple:
        num_scripts = args.multiple
    else:
        num_scripts = 1

    csv_file_paths = []
    ollama = initialize_ollama(use_ollama_host=args.use_ollama_host)

    ngrok_api_key = config.get("ngrok", "api_key", fallback=None)
    ngrok_auth_token = config.get("ngrok", "auth_token", fallback=None)
    openai_api_key = config.get("openai", "api_key", fallback=None)

    llm_choice = config.get("llm", "choice", fallback="ollama").lower()
    api_choice = config.get("api", "choice", fallback="openai").lower()

    if api_choice == "openai":
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif api_choice == "ollama":
        pass
    elif api_choice == "lmstudio":
        pass

    if not args.ranking or args.multiple:
        existing_csv_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv') and overall_goal in f and any(greek_alpha in f for greek_alpha in greek_alphabets)]
        existing_indices = [greek_alphabets.index(greek_alpha) for f in existing_csv_files for greek_alpha in greek_alphabets if greek_alpha in f]
        starting_index = max(existing_indices) + 1 if existing_indices else 0

        for i in range(starting_index, starting_index + num_scripts):
            logger.info(f"Starting script generation {i + 1} of {num_scripts} for the goal: '{overall_goal}'")

            if args.verbose:
                logger.info("Sending request to Ollama for agent data...")
            response = get_agent_data(ollama, overall_goal, delimiter=',')
            if not response:
                raise ValueError('No response from Ollama')

            file_path = save_csv_output(response, overall_goal, greek_alphabets)
            csv_file_paths.append(file_path)

            agents_data = parse_csv_data(response, delimiter=',', filename=file_path)
            if not agents_data:
                raise ValueError('No agent data parsed')

            file_name = os.path.basename(file_path).replace('.csv', '.py')
            crewai_script_path = os.path.join(os.getcwd(), file_name)
            crew_tasks = generate_crew_tasks(agents_data)

            write_crewai_script(agents_data, crew_tasks, crewai_script_path, args.use_ollama_host)
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
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
