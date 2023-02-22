from tensorflow import keras
import tensorflow as tf

def make_optimizer(learning_rate=0.0002, beta_1=0.5, beta_2=0.9):
    return keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5,
        beta_2=0.9
    )

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)
