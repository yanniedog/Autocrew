# ngrok-client.py

import os
import subprocess
from pathlib import Path 
import configparser
from pyngrok import ngrok

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
   
def validate_ngrok():
    try: 
        tunnel = ngrok.connect(80, bind_tls=True)
        tunnel.public_url 
        tunnel.stop()
    except:
        return False
        
    return True
    
def install_ngrok(auth_token): 
    ngrok.set_auth_token(auth_token)
    
    if not os.path.exists('/usr/local/bin/ngrok'):
        os.system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
        os.system('unzip ngrok-stable-linux-amd64.zip')
        os.system('sudo mv ngrok /usr/local/bin') 
        os.system('rm ngrok-stable-linux-amd64.zip')

    if not validate_ngrok():
        print("Error validating ngrok install")
      
if __name__=='__main__':  
    auth_token = get_auth_token()
    if auth_token: 
        install_ngrok(auth_token)
    else:
        print("Valid auth token not found")
