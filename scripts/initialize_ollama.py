# Filename: initialize_ollama.py

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests

def initialize_ollama(model='openhermes', use_ollama_host=False):
    if use_ollama_host:
        ollama_base_url = os.getenv('OLLAMA_HOST')
        if not ollama_base_url:
            raise EnvironmentError("OLLAMA_HOST environment variable not set")

        # Verify the OLLAMA_HOST URL's validity
        try:
            response = requests.get(ollama_base_url)
            response.raise_for_status()
        except Exception as e:
            print(f"Error validating OLLAMA_HOST URL: {e}")
            raise

        return Ollama(base_url=ollama_base_url, model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    else:
        return Ollama(model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
