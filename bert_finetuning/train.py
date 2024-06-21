import yaml
import torch
from torch import nn
from torch.nn import NLLLoss
from torch.optim import AdamW
from sklearn.utils import compute_class_weight
import numpy as np
from model import BERT
from data_processing import load_data, preprocess_data, create_dataloader
from transformers import BertTokenizerFast
from datetime import datetime
from typing import Tuple, List, Union

from get_hyperparameters import DATA_PATH, BERT_MODEL_NAME, SAVE_MODEL_PATH, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE

TIME = datetime.now().strftime('%Y%m%d_%H%M%S')

def train(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, np.ndarray]:

    model.train()
    total_loss, total_preds = 0.0, []

    for step, batch in enumerate(dataloader):
        if step % 50 == 0 and step != 0:
            print(f'Batch {step} of {len(dataloader)}')

        batch = [entry.to(device) for entry in batch]
        sent_id, mask, labels = batch

        model.zero_grad()
        preds = model(sent_id, mask)
        loss = criterion(preds, labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

def main() -> None:

    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    device: str = config['device'] if torch.cuda.is_available() else 'cpu'
    versioned_path: str = SAVE_MODEL_PATH.replace(".pt", f"_{TIME}.pt")

    df = load_data(DATA_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    train_seq, train_mask, train_y, test_seq, test_mask, test_y = preprocess_data(df, tokenizer, MAX_LENGTH)
    train_dataloader = create_dataloader(train_seq, train_mask, train_y, BATCH_SIZE)

    model = BERT(BERT_MODEL_NAME)
    model = model.to(device)

    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(train_y.cpu().numpy()), y=train_y.cpu().numpy())
    weights = torch.tensor(class_wts, dtype=torch.float).to(device)
    criterion = NLLLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        train_loss, _ = train(model, train_dataloader, device, optimizer, criterion)
        print(f'Training loss: {train_loss:.3f}')

    torch.save(model.state_dict(), versioned_path)

if __name__ == "__main__":
    main()
