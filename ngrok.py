# filename: ngrok.py

import configparser
import requests
import logging

from logging_config import setup_logging

def get_ngrok_api_key(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config.get('AUTHENTICATORS', 'ngrok_api_key')

def get_ngrok_tunnels(api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Ngrok-Version': '2'
    }
    response = requests.get('https://api.ngrok.com/tunnels', headers=headers)
    if response.status_code == 200:
        return response.json()['tunnels']
    else:
        raise Exception(f"Error retrieving tunnels: {response.text}")

def get_public_url(tunnels):
    for tunnel in tunnels:
        if tunnel['proto'] == 'https':
            return tunnel['public_url']
    return None

def get_colab_notebook_url(repo_api_url):
    response = requests.get(repo_api_url)
    if response.status_code == 200:
        repo_content = response.json()
        notebook_file = next((item for item in repo_content if item['name'].endswith('.ipynb')), None)
        if notebook_file:
            return f"https://colab.research.google.com/github/{notebook_file['path']}"
    raise Exception("Failed to fetch the Jupyter Notebook URL.")

def display_ngrok_setup_instructions(notebook_url):
    instructions = f"""
    NGROK TUNNEL SETUP INSTRUCTIONS:
    1. Visit the link to open the Jupyter Notebook in Google Colab: {notebook_url}
    2. Make a copy of the notebook in your Google Drive.
    3. In the Colab notebook, go to the 'Secrets' section.
    4. Create a new secret with the key 'authtoken' and paste your ngrok autotoken.
    5. Run the notebook to start the ngrok tunnel.
    Ensure your ngrok autotoken is correctly inserted and the notebook is executed.
    """
    print(instructions)

def main():
    setup_logging()  # Assuming you have a logging setup function
    try:
        ngrok_api_key = get_ngrok_api_key()
        tunnels = get_ngrok_tunnels(ngrok_api_key)
        public_url = get_public_url(tunnels)
        if public_url:
            logging.info(f"Ngrok public URL: {public_url}")
        else:
            logging.info("No public HTTPS tunnels found.")
            repo_api_url = "https://api.github.com/repos/yanniedog/Autocrew/contents/"
            notebook_url = get_colab_notebook_url(repo_api_url)
            display_ngrok_setup_instructions(notebook_url)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
