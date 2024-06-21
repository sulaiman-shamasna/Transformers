import torch
from torch import nn
from transformers import AutoModel
from get_hyperparameters import BERT_MODEL_NAME
from typing import Tuple

class BERT(nn.Module):
    def __init__(self, bert_model_name: str = BERT_MODEL_NAME) -> None:

        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name, return_dict=False)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def load_model(model_path: str, device: torch.device, bert_model_name: str = BERT_MODEL_NAME) -> BERT:

    model = BERT(bert_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model
