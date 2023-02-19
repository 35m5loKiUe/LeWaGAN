import tensorflow as tf
from LeWaGAN.interface.params import BATCH_SIZE, DATA_LOCATION, IMAGE_SIZE, ROOT_PATH, DATA_PATH
import os

def load_dataset():
    if DATA_LOCATION =='LOCAL':
        return tf.keras.utils.image_dataset_from_directory(
            os.path.join(ROOT_PATH, DATA_PATH),
            label_mode=None,
            color_mode='rgb',
            batch_size=BATCH_SIZE,
            image_size=(IMAGE_SIZE,IMAGE_SIZE)
        )

def normalize_dataset(dataset):
    normalization_layer = tf.keras.layers.Rescaling(scale=1./255)
    return dataset.map(lambda x: (normalization_layer(x)))
