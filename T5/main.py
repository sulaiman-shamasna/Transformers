from config import (
    MODEL,
    BATCH_SIZE,
    NUM_PROCS,
    EPOCHS,
    OUT_DIR,
    MAX_LENGTH,
    TRAIN_FILE,
    VALID_FILE,
    LEARNING_RATE
)
from data_processing import load_and_preprocess_data
from model_setup import setup_model
from training import train_model

def main():
    tokenized_train, tokenized_valid, tokenizer = load_and_preprocess_data(
        TRAIN_FILE, VALID_FILE, MODEL, MAX_LENGTH, NUM_PROCS
    )

    model, device = setup_model(MODEL)

    trainer = train_model(
        model, tokenized_train, tokenized_valid, OUT_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE
    )

    tokenizer.save_pretrained(OUT_DIR)

if __name__ == '__main__':
    main()
