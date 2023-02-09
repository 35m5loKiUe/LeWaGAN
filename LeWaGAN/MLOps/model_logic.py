from LeWaGAN.MLOps.params import EPOCHS, BATCH_SIZE, GENERATOR_OPTIMIZER, DISCRIMINATOR_OPTIMIZER, NOISE_DIM
from tensorflow.keras import losses, ones_like, zeros_like
from tensorflow import GradientTape, random


# The training loop begins with generator receiving a random seed as input.
# That seed is used to produce an image.
# The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator).
# The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.


def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(ones_like(real_output), real_output)
    fake_loss = cross_entropy(zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, cross_entropy):
    cross_entropy = losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(ones_like(fake_output), fake_output)

@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = random.normal([BATCH_SIZE, NOISE_DIM])

    with GradientTape() as gen_tape, GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return generator_optimizer, discriminator_optimizer
