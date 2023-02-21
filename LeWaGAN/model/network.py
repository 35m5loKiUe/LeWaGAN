import tensorflow as tf
from tensorflow import keras
from keras import layers

from LeWaGAN.Interface.params import NB_FILTERS, NOISE_DIM, IMAGE_SIZE

def make_generator_model():
    model = tf.keras.Sequential()

    # Dense layer to get sufficient dimensions to create 1st convolution
    model.add(layers.Dense(8 * 8 * NB_FILTERS, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape for 1st convolution
    model.add(layers.Reshape((8, 8, NB_FILTERS)))

    #Transpose convolution layers
    i = 16
    while i <= IMAGE_SIZE:
        model.add(layers.Conv2DTranspose(i, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        i = 2*i

    #Last convolution layer to return
    model.add(layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=None))
    model.add(layers.Activation('sigmoid'))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()

    #Convolution layers
    i = IMAGE_SIZE
    while i <= NB_FILTERS:
        model.add(layers.Conv2D(i, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        i = 2*i

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    return model
