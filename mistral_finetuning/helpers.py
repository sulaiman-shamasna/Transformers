import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    _ = load_dotenv(find_dotenv())

def get_mistral_api_key():
    load_env()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    return mistral_api_key
