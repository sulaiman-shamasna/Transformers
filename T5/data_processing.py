from datasets import load_dataset
from transformers import T5Tokenizer

def load_and_preprocess_data(train_file, valid_file, model_name, max_length, num_procs):
    dataset_train = load_dataset('csv', data_files=train_file, split='train')
    dataset_valid = load_dataset('csv', data_files=valid_file, split='train')

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = [f"assign tag: {title} {body}" for (title, body) in zip(examples['Title'], examples['Body'])]
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')

        cleaned_tag = [' '.join(''.join(tag.split('<')).split('>')[:-1]) for tag in examples['Tags']]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(cleaned_tag, max_length=max_length, truncation=True, padding='max_length')

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = dataset_train.map(preprocess_function, batched=True, num_proc=num_procs)
    tokenized_valid = dataset_valid.map(preprocess_function, batched=True, num_proc=num_procs)

    return tokenized_train, tokenized_valid, tokenizer
