# Filename: ngrok_client.py

import os
import subprocess

def install_ngrok():
    # Check if ngrok is already installed
    if not os.path.exists('/usr/local/bin/ngrok'):
        # Download ngrok
        os.system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
        # Unzip ngrok
        os.system('unzip ngrok-stable-linux-amd64.zip')
        # Move ngrok to /usr/local/bin
        os.system('sudo mv ngrok /usr/local/bin')
        # Remove the downloaded zip file
        os.system('rm ngrok-stable-linux-amd64.zip')

    # Validate ngrok's functionality by starting a test tunnel
    try:
        tunnel_process = subprocess.Popen(['ngrok', 'http', '80'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = tunnel_process.communicate()
        if error:
            raise Exception(f"Error starting ngrok test tunnel: {error}")

        tunnel_process.terminate()
    except Exception as e:
        print(f"Error validating ngrok functionality: {e}")
        raise

if __name__ == "__main__":
    install_ngrok()
