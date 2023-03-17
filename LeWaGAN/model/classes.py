import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np
from LeWaGAN.model.registery import save_model

from LeWaGAN.interface.params import MODEL_SAVE_FRQ, ROOT_PATH, IMG_STEPS_PATH, BATCH_SIZE

class WGAN(keras.Model):
    def __init__(
        self,
        image_size,
        nb_filters,
        noise_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
        dropout_rate=0.2,
        leaky_relu_slope=0.2,
        initial_learning_rate = 2e-3,
        decay_steps=2000,
        decay_rate=0.99,
        beta_1=0.5
    ):
        super().__init__()
        self.image_size = image_size
        self.nb_filters = nb_filters
        self.noise_dim = noise_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.dropout_rate=dropout_rate
        self.leaky_relu_slope=leaky_relu_slope
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps=decay_steps
        self.decay_rate=decay_rate
        self.beta_1=beta_1
        self.depth = int(np.log2(image_size/8))
        self.discriminator = self.discriminator()
        self.generator = self.generator()
        self.generator.summary()
        self.discriminator.summary()

    def discriminator(self):
        image_input = keras.Input(shape=(self.image_size, self.image_size, 3))
        x = image_input
        for _ in range(self.depth+1):
            x = layers.Conv2D(
                self.nb_filters, kernel_size=4, strides=2, padding="same", use_bias=False,
            )(x)
            x = layers.LeakyReLU(alpha=self.leaky_relu_slope)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        output_score = layers.Dense(1)(x)

        return keras.Model(image_input, output_score, name="discriminator")

    def generator(self):
        noise_input = keras.Input(shape=(self.noise_dim,))
        x = layers.Dense(4 * 4 * self.nb_filters, use_bias=False)(noise_input)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)
        x = layers.Reshape(target_shape=(4, 4, self.nb_filters))(x)
        for _ in range(self.depth):
            x = layers.Conv2DTranspose(
                self.nb_filters, kernel_size=4, strides=2, padding="same", use_bias=False,
            )(x)
            x = layers.BatchNormalization(scale=False)(x)
            x = layers.ReLU()(x)
        image_output = layers.Conv2DTranspose(
            3, kernel_size=4, strides=2, padding="same", activation="tanh",
        )(x)

        return keras.Model(noise_input, image_output, name="generator")

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
    # "hard sigmoid", useful for binary accuracy calculation from logits
    def step(self, values):
    # negative values -> 0.0, positive values -> 1.0
        return 0.5 * (1.0 + tf.sign(values))

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.noise_dim)
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
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))
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
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)
        self.real_accuracy.update_state(1.0, self.step(real_logits))
        self.generated_accuracy.update_state(0.0, self.step(fake_logits))

        return {m.name: m.result() for m in self.metrics[:-1]}

    def compile(self, **kwargs):
        super().compile(**kwargs)
        #lr_schedule=self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True
            )
        # separate optimizers for the two networks
        self.generator_optimizer = keras.optimizers.Adam(lr_schedule, self.beta_1)
        self.discriminator_optimizer = keras.optimizers.Adam(lr_schedule, self.beta_1)

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
            self.augmentation_probability_tracker
        ]

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, sqr_size=5, latent_dim=128, latent_vec=tf.random.normal(shape=(1,128))):
        self.sqr_size = sqr_size
        self.latent_dim = latent_dim
        self.latent_vec = latent_vec
        self.img_count = 0
        self.img_total_count = 0
        self.epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        self.img_count += BATCH_SIZE
        self.img_total_count += BATCH_SIZE
        if self.img_count > 3000:
            self.img_count = 0
            generated_images = self.model.generator(self.latent_vec)
            generated_images.numpy()
            fig = plt.figure(figsize=(10, 10))
            for i in range(generated_images.shape[0]):
                plt.subplot(self.sqr_size, self.sqr_size, i+1)
                img = keras.utils.array_to_img(generated_images[i])
                plt.imshow(img)
                plt.axis('off')
            plt.savefig(os.path.join(ROOT_PATH, IMG_STEPS_PATH,'{}_pictures_learnt_.png'.format(self.img_total_count)))
            print('\nGenerated samples after {} images learnt saved as {}'.format(self.img_total_count, os.path.join(ROOT_PATH, IMG_STEPS_PATH,'{}_pictures_learnt_.png\n'.format(self.img_total_count))))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch+1
        if not (epoch+1) % MODEL_SAVE_FRQ :
            save_model(self.model)

class GAN_ADA(keras.Model):
    def __init__(
        self,
        image_size,
        nb_filters,
        noise_dim,
        dropout_rate=0.2,
        ema=0.99,
        leaky_relu_slope=0.2,
        initial_learning_rate = 2e-4,
        decay_steps=4000,
        decay_rate=0.5,
        beta_1=0.5
        ):
        super().__init__()
        self.image_size = image_size
        self.nb_filters = nb_filters
        self.noise_dim = noise_dim
        self.ema = ema
        self.dropout_rate = dropout_rate
        self.leaky_relu_slope = leaky_relu_slope
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps=decay_steps
        self.decay_rate=decay_rate
        self.beta_1 = beta_1
        self.depth = int(np.log2(image_size/8))
        self.discriminator = self.discriminator()
        self.generator = self.generator()
        self.generator.summary()
        self.discriminator.summary()
        self.ema_generator = keras.models.clone_model(self.generator)

    # "hard sigmoid", useful for binary accuracy calculation from logits
    def step(self, values):
    # negative values -> 0.0, positive values -> 1.0
        return 0.5 * (1.0 + tf.sign(values))

    def compile(self, **kwargs):
        super().compile(**kwargs)
        #lr_schedule=self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=False
            )

        # separate optimizers for the two networks
        self.generator_optimizer = keras.optimizers.Adam(lr_schedule, self.beta_1)
        self.discriminator_optimizer = keras.optimizers.Adam(lr_schedule, self.beta_1)

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
            self.augmentation_probability_tracker
        ]

    def generate(self, batch_size, training):
        latent_samples = tf.random.normal(shape=(batch_size, self.noise_dim))
        # use ema_generator during inference
        if training:
            generated_images = self.generator(latent_samples, training)
        else:
            generated_images = self.ema_generator(latent_samples, training)
        return generated_images

    def adversarial_loss(self, batch_size, real_logits, generated_logits):
        # this is usually called the non-saturating GAN loss

        real_labels = tf.ones(shape=(batch_size, 1))
        generated_labels = tf.zeros(shape=(batch_size, 1))

        # the generator tries to produce images that the discriminator considers as real
        generator_loss = keras.losses.binary_crossentropy(
            real_labels, generated_logits, from_logits=True
        )
        # the discriminator tries to determine if images are real or generated
        discriminator_loss = keras.losses.binary_crossentropy(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
            from_logits=True,
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)

    def train_step(self, real_images):

        # use persistent gradient tape because gradients will be calculated twice
        with tf.GradientTape(persistent=True) as tape:
            batch_size = tf.shape(real_images)[0]
            generated_images = self.generate(batch_size, training=True)

            # separate forward passes for the real and generated images, meaning
            # that batch normalization is applied separately
            real_logits = self.discriminator(real_images, training=True)
            generated_logits = self.discriminator(generated_images, training=True)

            generator_loss, discriminator_loss = self.adversarial_loss(
                batch_size,
                real_logits,
                generated_logits
            )

        # calculate gradients and update weights
        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_weights
        )
        discriminator_gradients = tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_weights)
        )

        # update the augmentation probability based on the discriminator's performance

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy.update_state(1.0, self.step(real_logits))
        self.generated_accuracy.update_state(0.0, self.step(generated_logits))

        # track the exponential moving average of the generator's weights to decrease
        # variance in the generation quality
        for weight, ema_weight in zip(
            self.generator.weights, self.ema_generator.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    # DCGAN generator
    def generator(self):
        noise_input = keras.Input(shape=(self.noise_dim,))
        x = layers.Dense(4 * 4 * self.nb_filters, use_bias=False)(noise_input)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)
        x = layers.Reshape(target_shape=(4, 4, self.nb_filters))(x)
        for _ in range(self.depth):
            x = layers.Conv2DTranspose(
                self.nb_filters, kernel_size=4, strides=2, padding="same", use_bias=False,
            )(x)
            x = layers.BatchNormalization(scale=False)(x)
            x = layers.ReLU()(x)
        image_output = layers.Conv2DTranspose(
            3, kernel_size=4, strides=2, padding="same", activation="tanh",
        )(x)

        return keras.Model(noise_input, image_output, name="generator")

    # DCGAN discriminator
    def discriminator(self):
        image_input = keras.Input(shape=(self.image_size, self.image_size, 3))
        x = image_input
        for _ in range(self.depth+1):
            x = layers.Conv2D(
                self.nb_filters, kernel_size=4, strides=2, padding="same", use_bias=False,
            )(x)
            x = layers.BatchNormalization(scale=False)(x)
            x = layers.LeakyReLU(alpha=self.leaky_relu_slope)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        output_score = layers.Dense(1)(x)

        return keras.Model(image_input, output_score, name="discriminator")

class WGAN_upsampling(keras.Model):
    def __init__(
        self,
        image_size,
        nb_filters,
        noise_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
        dropout_rate=0.2,
        leaky_relu_slope=0.2,
        initial_learning_rate = 2e-4,
        decay_steps=4000,
        decay_rate=0.5,
        beta_1=0.5
    ):
        super().__init__()
        self.image_size = image_size
        self.nb_filters = nb_filters
        self.noise_dim = noise_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.dropout_rate=dropout_rate
        self.leaky_relu_slope=leaky_relu_slope
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps=decay_steps
        self.decay_rate=decay_rate
        self.beta_1=beta_1
        self.depth = int(np.log2(image_size/8))
        self.discriminator = self.discriminator()
        self.generator = self.generator()
        self.generator.summary()
        self.discriminator.summary()

    def discriminator(self):
        image_input = keras.Input(shape=(self.image_size, self.image_size, 3))
        x = image_input
        for _ in range(self.depth+1):
            x = layers.Conv2D(
                self.nb_filters,
                kernel_size=4,
                strides=2,
                padding="same",
                use_bias=False,
                )(x)

            x = layers.LeakyReLU(alpha=self.leaky_relu_slope)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        output_score = layers.Dense(1)(x)

        return keras.Model(image_input, output_score, name="discriminator")

    def generator(self):
        noise_input = keras.Input(shape=(self.noise_dim,))
        x = layers.Dense(4 * 4 * self.nb_filters, use_bias=False)(noise_input)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)
        x = layers.Reshape(target_shape=(4, 4, self.nb_filters))(x)
        for _ in range(self.depth):
            x = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(x)
            x = layers.Conv2D(self.nb_filters, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
            x = layers.BatchNormalization(scale=False)(x)
            x = layers.ReLU()(x)
        image_output = layers.Conv2DTranspose(
            3, kernel_size=4, strides=2, padding="same", activation="tanh",
        )(x)

        return keras.Model(noise_input, image_output, name="generator")

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
    # "hard sigmoid", useful for binary accuracy calculation from logits
    def step(self, values):
    # negative values -> 0.0, positive values -> 1.0
        return 0.5 * (1.0 + tf.sign(values))

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.noise_dim)
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
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.noise_dim))
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
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)
        self.real_accuracy.update_state(1.0, self.step(real_logits))
        self.generated_accuracy.update_state(0.0, self.step(fake_logits))

        return {m.name: m.result() for m in self.metrics[:-1]}

    def compile(self, **kwargs):
        super().compile(**kwargs)
        #lr_schedule=self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=False
            )
        # separate optimizers for the two networks
        self.generator_optimizer = keras.optimizers.Adam(lr_schedule, self.beta_1)
        self.discriminator_optimizer = keras.optimizers.Adam(lr_schedule, self.beta_1)

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
            self.augmentation_probability_tracker
        ]
