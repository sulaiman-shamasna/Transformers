import yaml

with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)


BERT_MODEL_NAME = config['bert_model_name']
DATA_PATH = config['data_path']
TRAINED_MODEL_PATH = config['trained_model_path']
SAVE_MODEL_PATH = config['model_save_path']

MAX_LENGTH = int(config['max_len'])
BATCH_SIZE = int(config['batch_size'])
EPOCHS = int(config['epochs'])
LEARNING_RATE = float(config['learning_rate'])