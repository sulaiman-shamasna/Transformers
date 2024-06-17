import numpy as np
import tensorflow_text as tf_text
import tensorflow as tf
import pathlib
from typing import Tuple, Callable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def download_and_extract_data(url: str, extract: bool = True) -> pathlib.Path:
    """
    Downloads and extracts the dataset from the given URL.
    
    Args:
        url (str): URL to the dataset.
        extract (bool): Whether to extract the dataset after downloading.
    
    Returns:
        pathlib.Path: Path to the extracted dataset file.
    """
    path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin=url, extract=extract)
    return pathlib.Path(path_to_zip).parent / 'spa-eng/spa.txt'

def load_data(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and processes the translation data from a given file path.
    
    Args:
        path (pathlib.Path): The path to the file containing the translation pairs.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays, one for the target language sentences and one for the context language sentences.
    """
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    context = np.array([context for target, context in pairs])
    target = np.array([target for target, context in pairs])

    return target, context

def tf_lower_and_split_punct(text: tf.Tensor) -> tf.Tensor:
    """
    Text standardization function to lower case and split punctuation.
    
    Args:
        text (tf.Tensor): Input text tensor.
    
    Returns:
        tf.Tensor: Processed text tensor.
    """
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def create_text_processor(vocab_size: int, dataset: tf.data.Dataset, map_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]) -> tf.keras.layers.TextVectorization:
    """
    Creates and adapts a TextVectorization layer.
    
    Args:
        vocab_size (int): Maximum vocabulary size.
        dataset (tf.data.Dataset): Dataset to adapt the processor on.
        map_func (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Function to map the dataset elements for the processor.
    
    Returns:
        tf.keras.layers.TextVectorization: Adapted TextVectorization layer.
    """
    text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=vocab_size,
        ragged=True)
    text_processor.adapt(dataset.map(map_func))
    return text_processor

def process_text(context: tf.Tensor, target: tf.Tensor, context_processor: tf.keras.layers.TextVectorization, target_processor: tf.keras.layers.TextVectorization) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Processes the text for input to the model.
    
    Args:
        context (tf.Tensor): Context sentences.
        target (tf.Tensor): Target sentences.
        context_processor (tf.keras.layers.TextVectorization): Processor for context sentences.
        target_processor (tf.keras.layers.TextVectorization): Processor for target sentences.
    
    Returns:
        Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]: Processed context and target tensors.
    """
    context = context_processor(context).to_tensor()
    target = target_processor(target)
    targ_in = target[:,:-1].to_tensor()
    targ_out = target[:,1:].to_tensor()
    return (context, targ_in), targ_out

def prepare_datasets(target: np.ndarray, context: np.ndarray, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares the training and validation datasets.
    
    Args:
        target (np.ndarray): Array of target sentences.
        context (np.ndarray): Array of context sentences.
        batch_size (int): Batch size for the datasets.
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    buffer_size = len(context)
    is_train = np.random.uniform(size=(len(target),)) < 0.8

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((context[is_train], target[is_train]))
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((context[~is_train], target[~is_train]))
        .shuffle(buffer_size)
        .batch(batch_size)
    )

    return train_raw, val_raw

def main() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.keras.layers.TextVectorization, tf.keras.layers.TextVectorization, np.ndarray, np.ndarray]:
    """
    Main function to prepare datasets and text processors.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.keras.layers.TextVectorization, tf.keras.layers.TextVectorization, np.ndarray, np.ndarray]:
        Training dataset, validation dataset, context text processor, target text processor, target raw data, context raw data.
    """
    path_to_file = download_and_extract_data('http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip')
    target_raw, context_raw = load_data(path_to_file)

    BATCH_SIZE = 64
    MAX_VOCAB_SIZE = 5000

    train_raw, val_raw = prepare_datasets(target_raw, context_raw, BATCH_SIZE)

    context_text_processor = create_text_processor(MAX_VOCAB_SIZE, train_raw, lambda context, target: context)
    target_text_processor = create_text_processor(MAX_VOCAB_SIZE, train_raw, lambda context, target: target)

    train_ds = train_raw.map(lambda context, target: process_text(context, target, context_text_processor, target_text_processor), tf.data.AUTOTUNE)
    val_ds = val_raw.map(lambda context, target: process_text(context, target, context_text_processor, target_text_processor), tf.data.AUTOTUNE)

    return train_ds, val_ds, context_text_processor, target_text_processor, target_raw, context_raw

# if __name__ == "__main__":
train_ds, val_ds, context_text_processor, target_text_processor, target_raw, context_raw = main()

for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
    print(ex_context_tok[0, :10].numpy()) 
    print()
    print(ex_tar_in[0, :10].numpy()) 
    print(ex_tar_out[0, :10].numpy())
