import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from typing import Tuple
from get_hyperparameters import DATA_PATH

def load_data(data_path: str) -> pd.DataFrame:
    
    return pd.read_csv(data_path)

def preprocess_data(df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    train_text, test_text, train_labels, test_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )
    
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(), max_length=max_len, padding='max_length', truncation=True, return_token_type_ids=False
    )
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(), max_length=max_len, padding='max_length', truncation=True, return_token_type_ids=False
    )
    
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())
    
    return train_seq, train_mask, train_y, test_seq, test_mask, test_y

def create_dataloader(seq: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> DataLoader:
    
    data = TensorDataset(seq, mask, y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader
