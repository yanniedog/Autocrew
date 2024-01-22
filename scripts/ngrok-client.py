import subprocess
import requests
import configparser
import os
import sys

def install_ngrok(auth_token):
    autocrew_dir = os.path.expanduser("~/autocrew")
    ngrok_path = os.path.join(autocrew_dir, "ngrok")
    if not os.path.exists(ngrok_path):
        print("Installing ngrok...")
        os.system(f'curl -s https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip -o {autocrew_dir}/ngrok.zip')
        os.system(f'unzip {autocrew_dir}/ngrok.zip -d {autocrew_dir}')
        os.system(f'rm {autocrew_dir}/ngrok.zip')
    os.system(f'{ngrok_path} authtoken {auth_token}')
    print("ngrok installed and configured successfully.")

def is_ngrok_installed():
    try:
        subprocess.run(["ngrok", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def get_ngrok_public_url(ngrok_api_key):
    ngrok_api_url = "https://api.ngrok.com/tunnels"
    headers = {
        "Authorization": f"Bearer {ngrok_api_key}",
        "Ngrok-Version": "2"
    }

    try:
        response = requests.get(ngrok_api_url, headers=headers)
        response.raise_for_status()
        tunnels = response.json().get("tunnels", [])
        
        if tunnels:
            public_url = tunnels[0].get("public_url")
            return public_url
        else:
            print("No active ngrok tunnels found.")
            return None

    except Exception as e:
        print(f"Error fetching ngrok tunnel information: {e}")
        return None

# Define the path to the config file
config_file_path = os.path.expanduser('~/autocrew/config.ini')

# Read configuration
config = configparser.ConfigParser()
try:
    config.read(config_file_path)
    ngrok_api_key = config.get('ngrok', 'api_key')
    ngrok_auth_token = config.get('ngrok', 'auth_token')
except configparser.NoSectionError:
    print("Error: 'ngrok' section not found in the config file.")
    sys.exit(1)
except configparser.NoOptionError as e:
    print(f"Error: Missing option in the config file - {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading the config file: {e}")
    sys.exit(1)

# Check if ngrok is installed
if not is_ngrok_installed():
    install_ngrok(ngrok_auth_token)

# Get the public URL
public_url = get_ngrok_public_url(ngrok_api_key)
if public_url:
    print("Ngrok Public URL:", public_url)
