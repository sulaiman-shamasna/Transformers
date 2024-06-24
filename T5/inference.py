import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def do_correction(text, model, tokenizer):
    input_text = f"assign tag: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=256,
        padding='max_length',
        truncation=True
    )

    corrected_ids = model.generate(
        inputs,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )

    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=True
    )
    return corrected_sentence

def process_inference_data(data_dir, model, tokenizer):
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r') as f:
            sentence = f.read()
            corrected_sentence = do_correction(sentence, model, tokenizer)
            print(f"QUERY: {sentence}\nTAGS: {corrected_sentence}")

def main():
    model_path = 'results_t5small/checkpoint-9000/'
    tokenizer_path = 'results_t5small'
    data_dir = 'inference_data/'

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    process_inference_data(data_dir, model, tokenizer)

if __name__ == '__main__':
    main()
