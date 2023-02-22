import os

# model params
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
NOISE_DIM = int(os.environ.get('NOISE_DIM'))
NB_FILTERS = int(os.environ.get('NB_FILTERS'))
NB_EIGENVECTORS = int(os.environ.get('NB_EIGENVECTORS'))

# Data locations

ROOT_PATH = os.environ.get('ROOT_PATH')

DATA_PATH = os.environ.get('DATA_PATH')
MODEL_PATH = os.environ.get('MODEL_PATH')

DATA_LOCATION = os.environ.get('DATA_LOCATION')
MODEL_LOCATION = os.environ.get('MODEL_LOCATION')

IMG_STEPS_PATH = os.environ.get('IMG_STEPS_PATH')

# registery params
MODEL_SAVE_FRQ = int(os.environ.get('MODEL_SAVE_FRQ'))
