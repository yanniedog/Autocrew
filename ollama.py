# filename: ollama.py

import subprocess
import os
import logging
import requests
import json
import configparser
import time

from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

CONFIG_FILE = 'config.ini'

CONFIG_FILE = 'config.ini'  # Path to the config.ini file

def read_config():
    """Read configuration from the config.ini file."""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def write_config(config):
    """Write the updated configuration back to the config.ini file."""
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

def update_config(section, option, value):
    config = configparser.ConfigParser()
    config.read('config.ini')
    config.set(section, option, value)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    logging.info(f"Updated config.ini: [{section}] {option} = {value}")

def update_ollama_config(model_name, ollama_host, use_remote=False):
    """Update the OLLAMA configuration in config.ini."""
    config = read_config()

    config['BASIC']['llm_endpoint'] = 'ollama'
    config['OLLAMA_CONFIG']['llm_model'] = model_name
    config['REMOTE_HOST_CONFIG']['use_remote_ollama_host'] = str(use_remote).lower()
    config['REMOTE_HOST_CONFIG']['ollama_host'] = ollama_host if use_remote else 'localhost:11434'

    write_config(config)

def initialize_ollama(use_remote_ollama_host, llm_model, ollama_host):
    connection_type = "remote" if use_remote_ollama_host else "local"
    logging.info(f"Initializing {connection_type} connection to Ollama using model {llm_model}...")

    # Start the Ollama service if it's not already running
    start_ollama_service()

    # Set default Ollama host if not specified in environment
    ollama_host = os.getenv('OLLAMA_HOST', ollama_host)
    logging.info(f"Ollama host: {ollama_host}")

    try:
        return Ollama(base_url=ollama_host, model=llm_model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    except Exception as e:
        logging.error(f"Failed to initialize Ollama: {e}")
        return None

def start_ollama_service():
    try:
        # Check if the Ollama service is running
        subprocess.check_output(["pgrep", "-f", "ollama serve"])
        logging.debug("Ollama service is already running.")
    except subprocess.CalledProcessError:
        # If the Ollama service is not running, start it
        logging.debug("Starting Ollama service...")
        subprocess.Popen(["ollama", "serve"], start_new_session=True)
        # Wait for 2 seconds to allow the service to start
        time.sleep(2)
        
def is_ollama_running():
    try:
        subprocess.check_output(["pgrep", "-f", "ollama serve"])
        logging.debug("Ollama service is already running.")
        return True
    except subprocess.CalledProcessError:
        return False

def format_size(bytes, suffix="B"):
    """Convert bytes to a more readable format in MB/s."""
    return f"{bytes / 1_000_000:.2f} MB{suffix}"


def pull_model(model_name, verbose=False):
    url = "http://localhost:11434/api/pull"
    headers = {"Content-Type": "application/json"}
    data = {"name": model_name, "stream": True}
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

    if verbose:
        pbar = None
        start_time = time.time()
        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode('utf-8'))

                if 'total' in decoded_line and 'completed' in decoded_line:
                    total = round(decoded_line['total'] / 1_000_000_000, 3)  # Convert to gigabytes and round to 3 decimal places
                    completed = round(decoded_line['completed'] / 1_000_000_000, 3)  # Convert to gigabytes and round to 3 decimal places

                    if pbar is None:
                        pbar = tqdm(total=total, dynamic_ncols=True, unit='gb', desc=f"Downloading {model_name}",
                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{unit}, {rate_fmt}, dur {elapsed}, eta {remaining}")

                    pbar.update(completed - pbar.n)  # Update with the completed amount in gigabytes

                if 'status' in decoded_line and decoded_line['status'] == 'success':
                    if pbar is not None:
                        pbar.close()
                    print("Model download completed successfully.")
                    return {"status": "success"}
        if pbar is not None:
            pbar.close()
    else:
        json_response = response.json()
        if json_response is not None:
            return json_response
        else:
            return {"status": "error", "message": "Invalid response from server"}







def list_models():
    url = "http://localhost:11434/api/tags"
    response = requests.get(url)
    return response.json()

def get_user_choice(prompt, num_options):
    while True:
        print(prompt)
        choice = input("Your choice (type 'back' to go back): ").strip().lower()
        if choice == 'back':
            return None
        if choice.isdigit() and 1 <= int(choice) <= num_options:
            return int(choice)
        print("Invalid choice. Please enter a number from the list or type 'back'.")

def scrape_and_list_urls(base_url):
    try:
        print("Requesting the webpage...")
        response = requests.get(base_url)
        response.raise_for_status()
        print("Webpage accessed successfully.")

        soup = BeautifulSoup(response.content, 'html.parser')
        all_links = soup.find_all('a', href=True)
        print(f"Total links found: {len(all_links)}")

        library_links = [a['href'] for a in all_links if a['href'].startswith('/library/')]
        print(f"Filtered links: {len(library_links)}")

        trimmed_links = sorted([link.replace('/library/', '') for link in library_links])
        print("Trimmed and sorted links:")

        for idx, link in enumerate(trimmed_links, 1):
            print(f"{idx}. {link}")

        choice = get_user_choice("Which model would you like to download?", len(trimmed_links))
        if choice is None:
            return None

        selected_model = trimmed_links[choice - 1]
        print(f"You selected: {selected_model}")

        model_url = f"{base_url}{selected_model}/tags"
        print(f"Formulated URL: {model_url}")

        if ollama_run_strings := scrape_ollama_run_strings(model_url):
            if ollama_model_to_pull := select_ollama_run_string(
                ollama_run_strings
            ):
                print(f"You selected: {ollama_model_to_pull}")
                return ollama_model_to_pull
            else:
                return None
        else:
            print("No 'ollama run' strings found on the model page.")
            return None

    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return None

def scrape_ollama_run_strings(model_url):
    try:
        print("Requesting the model webpage...")
        response = requests.get(model_url)
        response.raise_for_status()
        print("Model webpage accessed successfully.")

        soup = BeautifulSoup(response.content, 'html.parser')
        command_inputs = soup.find_all('input', class_='command')
        ollama_run_strings = [input_element['value'] for input_element in command_inputs]
        ollama_run_strings = sorted(set(ollama_run_strings))

        return ollama_run_strings
    
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return []

def select_ollama_run_string(ollama_run_strings):
    if not ollama_run_strings:
        return None

    # Remove the "ollama run " part from each string
    display_strings = [run_string.replace('ollama run ', '') for run_string in ollama_run_strings]

    print("Available model strings:")
    for idx, display_string in enumerate(display_strings, 1):
        print(f"{idx}. {display_string}")

    choice = get_user_choice("Please select the quantisation to use", len(display_strings))
    if choice is None:
        return None

    return ollama_run_strings[choice - 1].replace('ollama run ', '')


def update_config(section, option, value):
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    config.read(config_path)
    config.set(section, option, value)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def main():
    # Ensure the Ollama service is running before proceeding
    if not is_ollama_running():
        start_ollama_service()

    # Update ollama_host in config if it's a local connection
    ollama_host = "localhost:11434"
    update_config('REMOTE_HOST_CONFIG', 'ollama_host', ollama_host)

    while True:
        models = list_models()
        if 'models' in models:
            print("\nYour downloaded models:\n")
            for idx, model in enumerate(models['models'], 1):
                print(f"{idx}. {model['name']}")
            print(f"{len(models['models']) + 1}. [Download a NEW model]")
        else:
            print("Failed to list models.")
            return None

        choice = get_user_choice("\nEnter the number of the model to download, or type a model name:", len(models['models']) + 1)
        if choice is None:
            continue

        if choice <= len(models['models']):
            model_name = models['models'][choice - 1]['name']
            print(f"You have selected the model: {model_name}")
            # Update model in config
            update_config('OLLAMA_CONFIG', 'llm_model', model_name)
            return model_name

        elif choice == len(models['models']) + 1:
            base_url = "https://ollama.ai/library/"
            if selected_model := scrape_and_list_urls(base_url):
                model_name = selected_model
                print(f"Attempting to download model: {model_name}...")
                result = pull_model(model_name, verbose=True)

                if isinstance(result, dict) and 'status' in result and result['status'] == 'success':
                    print(f"Model {model_name} downloaded successfully.")
                    # Update model in config
                    update_config('OLLAMA_CONFIG', 'llm_model', model_name)
                    return model_name
                else:
                    print("Model download failed:", result)
                    return None
            else:
                continue
        else:
            print("Invalid choice. Please select a valid option.")
            continue

if __name__ == "__main__":
    main()
