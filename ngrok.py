# filename: ngrok.py

import configparser
import requests
import logging

from logging_config import setup_logging

# Function to read ngrok API key from config.ini
def get_ngrok_api_key(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config.get('AUTHENTICATORS', 'ngrok_api_key')

# Function to get the ngrok tunnel information using the ngrok API
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

# Function to extract the public URL from the tunnel information
def get_public_url(tunnels):
    return next(
        (
            tunnel['public_url']
            for tunnel in tunnels
            if tunnel['proto'] == 'https'
        ),
        None,
    )

def get_colab_notebook_url(repo_api_url):
    # New function to fetch Jupyter Notebook link from GitHub
    response = requests.get(repo_api_url)
    if response.status_code == 200:
        repo_content = response.json()
        if notebook_file := next(
            (item for item in repo_content if item['name'].endswith('.ipynb')),
            None,
        ):
            return f"https://colab.research.google.com/github/{notebook_file['path']}"
    raise Exception("Failed to fetch the Jupyter Notebook URL.")

def display_ngrok_setup_instructions(notebook_url):
    # New function to display setup instructions
    instructions = f"""
    NGROK TUNNEL SETUP INSTRUCTIONS:

    1. Visit the following link to open the Jupyter Notebook in Google Colab:
       \n{notebook_url}
    2. Make a copy of the notebook in your Google Drive.
    3. In the Colab notebook, go to the 'Secrets' section.
    4. Create a new secret with the key 'authtoken' and paste your ngrok autotoken as the value.
    5. Run the notebook to start the ngrok tunnel.
    Ensure your ngrok autotoken is correctly inserted and the notebook is executed.
    """
    print(instructions)

# Main function to execute the script
def main():
    setup_logging()  # Assuming you have a logging setup function
    config = configparser.ConfigParser()
    config.read('config.ini')

    try:
        ngrok_api_key = get_ngrok_api_key()
        tunnels = get_ngrok_tunnels(ngrok_api_key)
        if public_url := get_public_url(tunnels):
            logging.info(f"Ngrok public URL: {public_url}")
            # Store the public URL in the config.ini file
            config['REMOTE_HOST_CONFIG']['ollama_host'] = public_url
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
        else:
            logging.info("No public HTTPS tunnels found. Displaying setup instructions.")

            # Fetch and display instructions for setting up ngrok in Google Colab
            repo_api_url = "https://api.github.com/repos/yanniedog/Autocrew/contents/"
            notebook_url = get_colab_notebook_url(repo_api_url)
            display_ngrok_setup_instructions(notebook_url)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()