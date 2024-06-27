import sys
import pandas as pd
from datasets import Dataset
from data_preparation import tokenize_dataset
from model_utils import load_model_and_tokenizer, prepare_model_for_training, save_model, print_trainable_parameters
from training import fine_tune_model
from inference import perform_inference

# Check NumPy version
import numpy as np
print(f"NumPy version: {np.__version__}")

# Example DataFrame (replace this with your actual DataFrame)
df_faq = pd.DataFrame({
    'question': ['What is AI?', 'How to train a model?'],
    'answer': ['AI is the simulation of human intelligence in machines.', 'Training a model involves feeding it data and adjusting its parameters.']
})

# Create Dataset from DataFrame
data = Dataset.from_pandas(df_faq[['question', 'answer']])

# Load the model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"
model, tokenizer = load_model_and_tokenizer(model_name)

# Tokenize and prepare data
data = tokenize_dataset(data, tokenizer)

# Prepare model for fine-tuning
model = prepare_model_for_training(model)
print_trainable_parameters(model)

# Fine-tune the model
training_args = {
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "save_total_limit": 4,
    "logging_steps": 25,
    "output_dir": "output_dir",  # give the location where you want to store checkpoints
    "save_strategy": 'epoch',
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": 'cosine',
    "warmup_ratio": 0.05,
}

fine_tune_model(model, tokenizer, data, training_args)

# Save the fine-tuned model
save_model(model, 'trained_models/')

# Perform inference
prompt = "How can I use BDB Data Science LAB?"
response = perform_inference('trained_models/', prompt)
print(response)
