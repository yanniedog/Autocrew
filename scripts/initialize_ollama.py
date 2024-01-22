# Filename: initialize_ollama.py

import os  # Import the os module

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def initialize_ollama(model='openhermes', use_ollama_host=False, ollama_base_url=None):
    if ollama_base_url:
        return Ollama(base_url=ollama_base_url, model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    elif use_ollama_host:
        ollama_host = os.getenv('OLLAMA_HOST')
        if not ollama_host:
            raise EnvironmentError("OLLAMA_HOST environment variable not set")
        return Ollama(base_url=ollama_host, model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    else:
        return Ollama(model=model, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
