
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from IPython import display

import tensorflow as tf
from tensorflow import ones_like, zeros_like
from tensorflow.keras import layers, losses, Sequential, optimizers
from tensorflow.train import Checkpoint

from LeWaGAN.MLOps.params import EPOCHS, BATCH_SIZE, CHECKPOINT_PATH, IMAGE_SIZE
from LeWaGAN.MLOps.model_logic import discriminator_loss, train_step

global generator_optimizer, discriminator_optimizer
generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

def make_generator_model(image_size=IMAGE_SIZE):
    model = Sequential()

    model.add(layers.Dense(8*8*IMAGE_SIZE, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, IMAGE_SIZE)))
    assert model.output_shape == (None, 8, 8, IMAGE_SIZE)  # Note: None is the batch size

    layer_size = IMAGE_SIZE
    while layer_size > 8 :
        model.add(layers.Conv2DTranspose(layer_size, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        layer_size = layer_size / 2

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model

def make_discriminator_model(image_size=IMAGE_SIZE):
    model = Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation= tf.nn.relu))
    model.add(layers.Dense(1))
    return model


def save_model(
    generator_optimizer = None,
    discriminator_optimizer = None,
    generator = None,
    discriminator = None
    ):
    checkpoint_prefix = os.path.join(CHECKPOINT_PATH, "ckpt")
    checkpoint.save(file_prefix = checkpoint_prefix)


def train(dataset, discriminator, generator):

  for epoch in range(EPOCHS):
    start = time.time()

    for image_batch in dataset :
      generator_optimizer, discriminator_optimizer = train_step(
          image_batch,
          generator,
          discriminator,
          generator_optimizer,
          discriminator_optimizer
          )


    # Save the model every 15 epochs
    checkpoint_prefix = os.path.join(CHECKPOINT_PATH, "ckpt")
    checkpoint = Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
