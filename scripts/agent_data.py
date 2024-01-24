# agent_data.py

import argparse
import csv
import io
import json
import os
import requests
import traceback
import configparser
import subprocess
import logging
from datetime import datetime
from packaging import version
from crewai import Agent, Crew, Process, Task
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Any, Dict, List

# Importing custom modules
from get_ngrok_public_url import get_ngrok_public_url
from initialize_ollama import initialize_ollama
from save_csv_output import save_csv_output
from parse_csv_data import parse_csv_data
from define_agent import define_agent
from define_task import define_task
from script_generation import generate_crew_tasks
from get_agent_data import instruction

# Setting up logging
import logging
import os
import configparser

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

autocrew_version = "1.3.4.4"

config = configparser.ConfigParser()
config.read(os.path.join(os.path.expanduser("~"), "autocrew", "config.ini"))

def get_task_var_name(role):
    return f'task_{role.replace(" ", "_").replace("-", "_").replace(".", "_")}'

def get_ngrok_public_url():
    try:
        # Correcting the path to ngrok-client.py
        ngrok_client_script_path = os.path.expanduser('~/autocrew/scripts/ngrok-client.py')
        process = subprocess.run(['/bin/python3', ngrok_client_script_path], capture_output=True, text=True)
        output = process.stdout.strip()
        if not output.startswith("https://"):
            raise ValueError("Invalid ngrok URL received")
        return output
    except Exception as e:
        logger.error(f"Error getting ngrok public URL: {e}")
        return None

def main():
    # Adding initial logging to confirm script execution
    logger.info("Starting agent_data.py script")
    logger.info(f"Autocrew (v{autocrew_version}) for CrewAI")
    
    parser = argparse.ArgumentParser(description="Run the agent data script")
    parser.add_argument("--use_ollama_host", help="Use Ollama host", action='store_true')
    parser.add_argument("--overall_goal", help="Specify the overall goal", type=str)
    parser.add_argument("--multiple", help="Specify the number of scripts to generate", type=int, default=1)
    parser.add_argument("--ranking", help="Specify the ranking", action='store_true')
    parser.add_argument("--verbose", help="Enable verbose output", action='store_true')
    args = parser.parse_args()

    if not any(vars(args).values()):
        logger.info("No arguments provided. Exiting script.")
        return

    greek_alphabets = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
                       "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]
    
    # Get the API key from the config.ini file
    api_key = config.get("openai", "api_key")

    # Get the ngrok URL
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
            response = get_agent_data(ollama, overall_goal, delimiter=',', ngrok_url=ngrok_url, api_key=api_key)
            if not response:
                raise ValueError('No response from Ollama')

            file_path = save_csv_output(response, overall_goal, greek_alphabets)
            csv_file_paths.append(file_path)

            agents_data = parse_csv_data(response, delimiter=',', filename=file_path)
            if not agents_data:
                raise ValueError('No agent data parsed')

            crew_tasks = generate_crew_tasks(agents_data)

            return crew_tasks, overall_goal, csv_file_paths, args
    response = get_agent_data(ollama, overall_goal, delimiter=',', ngrok_url=ngrok_url, api_key=api_key)
    if not response:
        raise ValueError('No response from Ollama')


def get_agent_data(ollama, overall_goal, delimiter, ngrok_url, api_key):
    logger.debug(f"Starting get_agent_data with arguments: {locals()}")
    if not ngrok_url.startswith("https://"):
        ngrok_url = "https://" + ngrok_url

    api_endpoint = ngrok_url

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "instruction": instruction,
        "overall_goal": overall_goal,
        "delimiter": delimiter
    }

    try:
        logger.info(f"Sending request to Ollama: URL: {api_endpoint}, Headers: {headers}, Payload: {payload}")
        response = requests.post(api_endpoint, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Ollama service responded with status code: {response.status_code}, Response: {response.text}")
            raise Exception(f"Ollama service responded with status code: {response.status_code}")
        else:
            logger.info(f"Response from Ollama received: {response.text}")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in communicating with Ollama: {e}")
        return None
    finally:
        logger.debug("Exiting get_agent_data")

if __name__ == "__main__":
    main()
