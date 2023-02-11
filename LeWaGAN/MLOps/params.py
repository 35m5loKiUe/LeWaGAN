import os

# model params
IMAGE_SIZE = os.environ.get('IMAGE_SIZE')
EPOCHS = os.environ.get('EPOCHS')
BATCH_SIZE = os.environ.get('BATCH_SIZE')
NOISE_DIM = os.environ.get('NOISE_DIM')

# Data locations
ROOT_PATH = os.environ.get('ROOT_PATH')
PREPROC_LOCATION = os.environ.get('PREPROC_LOCATION')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH')
DATA_PATH = os.environ.get('DATA_PATH')
PREPROC_PATH = os.environ.get('PREPROC_PATH')
