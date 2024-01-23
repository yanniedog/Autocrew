import os
import requests
import configparser
from pathlib import Path

from langchain_community.llms import Ollama
from get_ngrok_public_url import get_ngrok_public_url

CONFIG_FILE = os.path.join(Path.home(), "autocrew", "config.ini")

def get_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def validate_base_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return True
    except: 
        return False

def initialize_ollama(model='openhermes', use_ollama_host=False):
    base_url = "https://"

    if use_ollama_host:
        custom_host = os.getenv('OLLAMA_HOST')
        if not custom_host:
            print("Error - OLLAMA_HOST not set")
            return  

        base_url += custom_host

        if not validate_base_url(base_url):
            print(f"Invalid URL: {base_url}")
            return

    if not use_ollama_host or not base_url:
        ngrok_url = get_ngrok_public_url()
        if ngrok_url:
            base_url += ngrok_url
        else:
            print("Error - Invalid or no ngrok URL provided")
            return

    ollama = Ollama(base_url=base_url, model=model, verbose=True)
    
    return ollama

if __name__ == "__main__":
    ollama = initialize_ollama()