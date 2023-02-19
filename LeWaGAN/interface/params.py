import os

# model params
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
NOISE_DIM = int(os.environ.get('NOISE_DIM'))
NB_FILTERS = os.environ.get('NB_FILTERS')

# Data locations
ROOT_PATH = os.environ.get('ROOT_PATH')

DATA_PATH = os.environ.get('DATA_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')

DATA_LOCATION = os.environ.get('DATA_LOCATION')
MODEL_LOCATION = os.environ.get('MODEL_LOCATION')

# registery params
MODEL_SAVE_FRQ = os.environ.get('MODEL_SAVE_FRQ')
