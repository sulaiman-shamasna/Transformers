from transformers import AutoTokenizer
from datasets import Dataset

def gen_prompt(text_input):
    return f"""
    <human>: {text_input["question"]}
    <assistant>: {text_input["answer"]}
    """.strip()

def gen_and_tok_prompt(text_input, tokenizer):
    full_input = gen_prompt(text_input)
    tok_full_prompt = tokenizer(full_input, padding=True, truncation=True, return_tensors="pt")
    return {
        'input_ids': tok_full_prompt['input_ids'][0],
        'attention_mask': tok_full_prompt['attention_mask'][0]
    }

def tokenize_and_check_length(batch, tokenizer):
    tokenized_batch = [gen_and_tok_prompt(x, tokenizer) for x in batch]
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

def tokenize_dataset(data, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    return data.map(lambda x: tokenize_and_check_length([x], tokenizer), batched=True, remove_columns=["question", "answer"])
