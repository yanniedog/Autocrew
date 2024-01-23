
# ngrok-client.py

import os
from pyngrok import ngrok
import configparser
from pathlib import Path

CONFIG_PATH = os.path.join(Path.home(), "autocrew", "config.ini")

def get_config():
    config = configparser.ConfigParser()
    try:
        config.read(CONFIG_PATH) 
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
    except:
        print("Auth token not found in config")
        return None

def start_ngrok():
    try:
        tunnel = ngrok.connect(80, bind_tls=True)
        return tunnel.public_url
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        return None

def install_and_start_ngrok(auth_token):
    ngrok.set_auth_token(auth_token)

    if not os.path.exists('/usr/local/bin/ngrok'):
        os.system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
        os.system('unzip ngrok-stable-linux-amd64.zip')
        os.system('sudo mv ngrok /usr/local/bin')
        os.system('rm ngrok-stable-linux-amd64.zip')

    return start_ngrok()

if __name__ == '__main__':
    auth_token = get_auth_token()
    if auth_token:
        public_url = install_and_start_ngrok(auth_token)
        print(public_url)
    else:
        print("Valid auth token not found")
