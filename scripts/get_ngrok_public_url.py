# Filename: get_ngrok_public_url.py

import os
import subprocess

def get_ngrok_public_url():
    # Path to the ngrok-client.py script
    ngrok_client_script = os.path.expanduser('~/autocrew/scripts/ngrok-client.py')

    try:
        # Execute the script and capture its output
        result = subprocess.run(['python3', ngrok_client_script], capture_output=True, text=True, check=True)
        public_url = result.stdout.strip()
        # Set OLLAMA_HOST environment variable
        os.environ['OLLAMA_HOST'] = public_url
        return public_url
    except subprocess.CalledProcessError as e:
        print(f"Error running ngrok-client.py: {e}")
        return None
