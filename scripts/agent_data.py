# agent_data.py

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
import logging
from typing import Any, Dict, List
import subprocess

# Importing custom modules
from get_ngrok_public_url import get_ngrok_public_url
from initialize_ollama import initialize_ollama
from save_csv_output import save_csv_output
from parse_csv_data import parse_csv_data
from define_agent import define_agent
from get_task_var_name import get_task_var_name
from define_task import define_task
from generate_crew_tasks import generate_crew_tasks

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

autocrew_version = "1.3.4"

def main():
    parser = argparse.ArgumentParser(description="Run the agent data script")
    parser.add_argument("--use_ollama_host", help="Use Ollama host", action='store_true')
    parser.add_argument("--overall_goal", help="Specify the overall goal", type=str)
    parser.add_argument("--multiple", help="Specify the number of scripts to generate", type=int, default=1)
    parser.add_argument("--ranking", help="Specify the ranking", action='store_true')
    parser.add_argument("--verbose", help="Enable verbose output", action='store_true')
    args = parser.parse_args()

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

    logger.info("To see the available command line parameters, type: python3 agent_data.py --help")

    overall_goal = args.overall_goal or input('\nPlease specify the overall goal: \n')

    num_scripts = args.multiple

    csv_file_paths = []
    use_ollama_host = hasattr(args, 'use_ollama_host') and args.use_ollama_host
    ollama = initialize_ollama(use_ollama_host=use_ollama_host)

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

            crew_tasks = generate_crew_tasks(agents_data)

            return crew_tasks, overall_goal, csv_file_paths, args

def get_agent_data(ollama, overall_goal, delimiter):
    instruction = (
        f'Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with the goal: "{overall_goal}". '
        f'Use the delimiter "{delimiter}" to separate the fields. '
        'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
        'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
        'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
    )
    response = ollama.invoke(instruction.format(overall_goal=overall_goal, delimiter=delimiter))
    return response

if __name__ == "__main__":
    ollama = initialize_ollama()
    main_agent_data(ollama)
