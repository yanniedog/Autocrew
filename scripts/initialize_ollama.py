import os
from pathlib import Path 
import configparser
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks import CallbackManager, StreamingStdoutCallbackHandler
import requests
    
CONFIG_PATH = os.path.join(Path.home(), "autocrew", "config.ini")   

def get_config():
    config = configparser.ConfigParser()
    try:
        config.read(CONFIG_PATH)
    except Exception as e: 
        print(f"Error reading config file: {e}")
        return None
    
    return config

def initialize_ollama(model='openhermes', use_ollama_host=False):
    config = get_config()  
    api_key = None
    
    if config:
        try:
            api_key = config["openai"]["api_key"]  
        except Exception as e:
            print(f"Error getting OpenAI API key: {e}")
    
    callback_manager = CallbackManager([StreamingStdoutCallbackHandler()])
    
    if use_ollama_host:
        if "OLLAMA_HOST" not in os.environ:
            print("Error: OLLAMA_HOST env var not set")
            return
        
        base_url = os.environ["OLLAMA_HOST"]
        try:
            response = requests.get(base_url)
            response.raise_for_status() 
        except Exception as e:
            print(f"Error validating OLLAMA_HOST URL: {e}")
            return
       
        return Ollama(base_url=base_url, 
                    model=model,  
                    verbose=True,
                    api_key=api_key,
                    callback_manager=callback_manager)     
    else:
       return Ollama(model=model,
                    verbose=True, 
                    api_key=api_key,
                    callback_manager=callback_manager)
