import os

# model params
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
NOISE_DIM = int(os.environ.get('NOISE_DIM'))

# Data locations
ROOT_PATH = os.environ.get('ROOT_PATH')
PREPROC_LOCATION = os.environ.get('PREPROC_LOCATION')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH')
DATA_PATH = os.environ.get('DATA_PATH')
PREPROC_PATH = os.environ.get('PREPROC_PATH')
DATA_LOCATION = os.environ.get('DATA_LOCATION')
