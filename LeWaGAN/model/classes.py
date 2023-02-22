import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from LeWaGAN.model.registery import save_model

from LeWaGAN.interface.params import MODEL_SAVE_FRQ, ROOT_PATH, MODEL_PATH, IMG_STEPS_PATH

class WGAN(keras.Model):
    def __init__(
        self,
        image_size,
        nb_filters,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.nb_filters = nb_filters
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.discriminator = self.make_discriminator()
        self.generator = self.make_generator()
        self.g_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )
        self.d_optimizer = keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )

    def make_discriminator(self):
        discriminator = tf.keras.Sequential()
        i = self.image_size
        while i <= self.nb_filters:
            discriminator.add(layers.Conv2D(i, (3, 3), strides=(2, 2), padding='same'))
            discriminator.add(layers.LeakyReLU())
            discriminator.add(layers.Dropout(0.2))
            i = 2*i
        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(self.image_size))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.2))
        discriminator.add(layers.Dense(self.image_size/2))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.2))
        discriminator.add(layers.Dense(1))
        return discriminator

    def make_generator(self):
        generator = tf.keras.Sequential()
        # Dense layer to get sufficient dimensions to create 1st convolution
        generator.add(layers.Dense(8 * 8 * self.nb_filters, use_bias=False, input_shape=(100,)))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())
        # Reshape for 1st convolution
        generator.add(layers.Reshape((8, 8, self.nb_filters)))
        # upsampling layers to increase image size
        i = 16
        while i <= self.image_size:
            generator.add(layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
            generator.add(layers.Conv2D(i, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            generator.add(layers.BatchNormalization(momentum=0.9, epsilon=0.00001))
            generator.add(layers.LeakyReLU(alpha=0.2))
            i = 2*i
        #Last convolution layer to return a 3 channels (RGB) image
        generator.add(layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation=None))
        generator.add(layers.Activation('sigmoid'))
        return generator

    @tf.function
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal(
            [batch_size, 1, 1, 1],
            mean=0.0,
            stddev=1.0
        )
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    @tf.function
    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)
                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.discriminator_loss(real_logits, fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.generator_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, sqr_size=5, latent_dim=128, latent_vec=tf.random.normal(shape=(1,128))):
        self.sqr_size = sqr_size
        self.latent_dim = latent_dim
        self.latent_vec = latent_vec

    def on_epoch_end(self, epoch, logs=None):

        if not (epoch+1) % MODEL_SAVE_FRQ :
            save_model(self.model)

        generated_images = self.model.generator(self.latent_vec)
        generated_images.numpy()

        fig = plt.figure(figsize=(10, 10))
        for i in range(generated_images.shape[0]):
            plt.subplot(self.sqr_size, self.sqr_size, i+1)
            img = keras.utils.array_to_img(generated_images[i])
            plt.imshow(img)
            plt.axis('off')
        plt.savefig(os.path.join(ROOT_PATH, IMG_STEPS_PATH,'image_at_epoch_{:04d}.png'.format(epoch+1)))
        print('\nGenerated samples for epoch {:04d} saved as {}'.format(epoch+1, os.path.join(ROOT_PATH, IMG_STEPS_PATH,'image_at_epoch_{:04d}.png\n'.format(epoch+1))))
