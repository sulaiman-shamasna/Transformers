
import sys
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import pandas as pd
import transformers

# Check NumPy version
print(f"NumPy version: {np.__version__}")

# Example DataFrame (replace this with your actual DataFrame)
df_faq = pd.DataFrame({
    'question': ['What is AI?', 'How to train a model?'],
    'answer': ['AI is the simulation of human intelligence in machines.', 'Training a model involves feeding it data and adjusting its parameters.']
})

def gen_prompt(text_input):
    print(f"gen_prompt input: {text_input}")
    return f"""
    <human>: {text_input["question"]}
    <assistant>: {text_input["answer"]}
    """.strip()

def gen_and_tok_prompt(text_input):
    full_input = gen_prompt(text_input)
    tok_full_prompt = tokenizer(full_input, padding=True, truncation=True, return_tensors="pt")
    return {
        'input_ids': tok_full_prompt['input_ids'][0],
        'attention_mask': tok_full_prompt['attention_mask'][0]
    }

# Create Dataset from DataFrame
data = Dataset.from_pandas(df_faq[['question', 'answer']])

# Load the model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)

# Data tokenization
tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_check_length(batch):
    tokenized_batch = [gen_and_tok_prompt(x) for x in batch]
    input_ids_lengths = [len(x['input_ids']) for x in tokenized_batch]
    attention_mask_lengths = [len(x['attention_mask']) for x in tokenized_batch]

    # Ensure all elements have the same length
    assert all(length == input_ids_lengths[0] for length in input_ids_lengths), "Inconsistent lengths in input_ids"
    assert all(length == attention_mask_lengths[0] for length in attention_mask_lengths), "Inconsistent lengths in attention_mask"

    # Combine all input_ids and attention_mask into single tensors
    input_ids = [x['input_ids'].tolist() for x in tokenized_batch]
    attention_mask = [x['attention_mask'].tolist() for x in tokenized_batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

# Use a lambda that properly maps over the rows of the dataset
data = data.map(lambda x: tokenize_and_check_length([x]), batched=True, remove_columns=["question", "answer"])

print(f'Data type is {type(data)}')
# sys.exit(0)


# Prepare model for finetuning

# Step(1)
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Step(2)
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Step(3)
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
# from transformers import PeftConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Fine tune the model
training_args = transformers.TrainingArguments(
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=4,
    logging_steps=25,
    output_dir="output_dir", # give the location where you want to store checkpoints 
    save_strategy='epoch',
    optim="paged_adamw_8bit",
    lr_scheduler_type = 'cosine',
    warmup_ratio = 0.05,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# Save the finetuned model
model.save_pretrained('trained_models/')

# Inference
config = PeftConfig.from_pretrained("trained_models/")
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
#     load_in_8bit=True,
#     device_map='auto',
    trust_remote_code=True,

)



tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path)

model_inf = PeftModel.from_pretrained(model,"trained_models/" )

# create your own prompt  
prompt = f"""
    <human>: How can i use BDB Data Science LAB?
    <assistant>: 
    """.strip()

# encode the prompt 
encoding = tokenizer(prompt, return_tensors= "pt").to(model.device)

# set teh generation configuration params 
gen_config = model_inf.generation_config
gen_config.max_new_tokens = 200
gen_config.temperature = 0.2
gen_config.top_p = 0.7
gen_config.num_return_sequences = 1
gen_config.pad_token_id = tokenizer.eos_token_id
gen_config.eos_token_id = tokenizer.eos_token_id

print('BEFORE TOCKENIZER')
import torch
# do the inference 
with torch.inference_mode():
    outputs = model.generate(input_ids = encoding.input_ids, attention_mask = encoding.attention_mask,generation_config = gen_config )
print(tokenizer.decode(outputs[0], skip_special_tokens = True ))



