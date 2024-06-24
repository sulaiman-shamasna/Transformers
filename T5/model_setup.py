import torch
from transformers import T5ForConditionalGeneration

def setup_model(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("GPU is available. Using GPU.")
    else:
        print("GPU is not available. Using CPU.")

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} total parameters.")
    print(f"{total_trainable_params:,} training parameters.")

    return model, device
