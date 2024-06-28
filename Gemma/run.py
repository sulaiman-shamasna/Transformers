from transformers import AutoTokenizer, AutoModelForCausalLM

import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    _ = load_dotenv(find_dotenv())

def get_access_token():
    load_env()
    access_token = os.getenv("ACCESS_TOKEN")
    return access_token

access_token = get_access_token()

# Ensure the access token is correctly used for both tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", use_auth_token=access_token)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
