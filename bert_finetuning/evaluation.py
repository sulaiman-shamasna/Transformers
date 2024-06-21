import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from model import load_model
from data_processing import load_data, preprocess_data
from transformers import BertTokenizerFast
from get_hyperparameters import DATA_PATH, BERT_MODEL_NAME, TRAINED_MODEL_PATH

def main() -> None:

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len: int = 25

    df: pd.DataFrame = load_data(DATA_PATH)
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    _, _, _, test_seq, test_mask, test_y = preprocess_data(df, tokenizer, max_len)

    model = load_model(TRAINED_MODEL_PATH, device, BERT_MODEL_NAME)

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis=1)

    print(classification_report(test_y, preds))
    print(pd.crosstab(test_y, preds))

if __name__ == "__main__":
    main()
