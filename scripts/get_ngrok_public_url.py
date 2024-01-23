# get_ngrok_public_url.py
import os
import subprocess
from pathlib import Path  
import configparser

CONFIG_FILE = os.path.join(Path.home(), "autocrew", "config.ini")

def get_config():
    config = configparser.ConfigParser()
    try: 
        config.read(CONFIG_FILE)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None 
    return config

def get_auth_token():
    config = get_config()
    if not config:
        return None
    
    try: 
        return config["ngrok"]["auth_token"]
    except Exception as e:
        print("Auth token not found in config")
        return None
        

def run_ngrok_client(auth_token):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to ngrok-client.py
    ngrok_client_path = os.path.join(dir_path, 'ngrok-client.py')

    # Run the ngrok client with the Python interpreter
    result = subprocess.run(['python3', ngrok_client_path, auth_token], capture_output=True, text=True)
    public_url = result.stdout
    return public_url


def get_ngrok_public_url():
    config = get_config()  
    auth_token = get_auth_token()
    
    if auth_token is None:
        print("Valid auth token not found")
        return
        
    url = run_ngrok_client(auth_token)
    return url   
