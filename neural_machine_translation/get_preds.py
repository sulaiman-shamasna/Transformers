import tensorflow as tf
from models import Translator
import tensorflow_text as tf_text
from preprocessing import create_text_processor, tf_lower_and_split_punct, load_data, download_and_extract_data, prepare_datasets
from metrics import masked_loss, masked_acc
from models import Encoder, Decoder, CrossAttention

def load_pretrained_model(model_path: str, context_text_processor, target_text_processor):
    # Define custom objects for loading
    custom_objects = {
        'tf_lower_and_split_punct': tf_lower_and_split_punct,
        'masked_loss': masked_loss,
        'masked_acc': masked_acc,
        'Encoder': Encoder,
        'CrossAttention': CrossAttention,
        'Decoder': Decoder,
        'Translator': Translator,
    }

    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def prepare_text_processors():
    # Load the raw data
    path_to_file = download_and_extract_data('http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip')
    target_raw, context_raw = load_data(path_to_file)
    
    # Create text processors
    BATCH_SIZE = 64
    MAX_VOCAB_SIZE = 5000
    train_raw, val_raw = prepare_datasets(target_raw, context_raw, BATCH_SIZE)
    context_text_processor = create_text_processor(MAX_VOCAB_SIZE, train_raw, lambda context, target: context)
    target_text_processor = create_text_processor(MAX_VOCAB_SIZE, train_raw, lambda context, target: target)
    
    return context_text_processor, target_text_processor

def run_inference(model, inputs):
    results = model.translate(inputs)
    for i, input_text in enumerate(inputs):
        print(f"Input: {input_text}")
        print(f"Output: {results[i].numpy().decode()}\n")

def main():
    # Prepare text processors
    context_text_processor, target_text_processor = prepare_text_processors()
    
    # Load the pretrained model
    model_path = 'trained_models/big_model11'
    model = load_pretrained_model(model_path, context_text_processor, target_text_processor)
    
    # Inputs for inference
    inputs = [
        'Hace mucho frio aqui.',  # "It's really cold here."
        'Esta es mi vida.',       # "This is my life."
        'Su cuarto es un desastre.',  # "His room is a mess"
        'Hola.'                   # "Hello."
    ]
    
    # Run inference
    run_inference(model, inputs)

if __name__ == "__main__":
    main()
