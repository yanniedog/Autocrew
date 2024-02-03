import configparser
import requests

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
    for tunnel in tunnels:
        if tunnel['proto'] == 'https':  # Assuming you want the HTTPS URL
            return tunnel['public_url']
    return None

# Main function to execute the script
def main():
    try:
        ngrok_api_key = get_ngrok_api_key()
        tunnels = get_ngrok_tunnels(ngrok_api_key)
        public_url = get_public_url(tunnels)
        if public_url:
            print(f"Ngrok public URL: {public_url}")
        else:
            print("No public HTTPS tunnels found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
