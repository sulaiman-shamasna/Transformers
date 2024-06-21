import numpy as np
import torch
from transformers import BertTokenizerFast
from model import load_model
from get_hyperparameters import TRAINED_MODEL_PATH, BERT_MODEL_NAME
import yaml
from typing import Dict

def load_config(config_path: str) -> Dict:

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def main() -> None:

    config: Dict = load_config("config.yaml")
    
    trained_model_path: str = config['trained_model_path']
    bert_model_name: str = config['bert_model_name']

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    model = load_model(trained_model_path, device, bert_model_name)
    model = model.to(device)
    model.load_state_dict(torch.load(trained_model_path))

    max_sequence_length: int = 25

    samples = [
        "Hi Sulaiman, what time is your birthday party tomorrow?",
        "Hey, can you send me the report by EOD?",
        "Remember to pick up milk on your way home.",
        "Congratulations on your promotion!",
        "URGENT: Your account has been suspended. Click here to verify your details.",
        "Free Rolex watches! Limited time offer. Claim now!",
        "Make $10,000 in a week! Just click this link.",
        "You've won a lottery. Please provide your bank details to claim the prize.",
        "Meet hot singles in your area tonight!",
        "Click here to win a free vacation package."
    ]

    all_preds = []

    for sample in samples:
        with torch.no_grad():
            tokens = tokenizer.batch_encode_plus(
                [sample], max_length=max_sequence_length, padding='max_length', truncation=True, return_token_type_ids=False
            )
            sequence = torch.tensor(tokens['input_ids'])
            mask = torch.tensor(tokens['attention_mask'])
            preds = model(sequence.to(device), mask.to(device))
            preds = preds.detach().cpu().numpy()

        preds = np.argmax(preds, axis=1)
        all_preds.append(preds[0])

    print("\n=== Inference Results ===")
    for i, sample in enumerate(samples):
        result = "Spam" if all_preds[i] == 1 else "Real"
        print(f"Sample {i+1}: '{sample}' --> Prediction: {result}")

if __name__ == "__main__":
    main()
