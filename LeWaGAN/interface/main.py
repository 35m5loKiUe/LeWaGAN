from LeWaGAN.interface.params import IMAGE_SIZE, NB_FILTERS, NOISE_DIM, BATCH_SIZE, EPOCHS

from LeWaGAN.model.network import make_generator_model, make_discriminator_model
from LeWaGAN.model.classes import WGAN, GANMonitor
from LeWaGAN.model.solver import make_optimizer, discriminator_loss, generator_loss

from LeWaGAN.data.dataset import load_dataset, normalize_dataset

from LeWaGAN.postprocessing.generate import image_with_eigenvectors, generate_noise

def generate_model():
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    model = WGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=NOISE_DIM,
        discriminator_extra_steps=3
        )
    generator_optimizer = make_optimizer()
    discriminator_optimizer = make_optimizer()
    model.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss
        )
    return model

def train_model(model, dataset):
    latent_vec=generate_noise(5**2)
    cbk = GANMonitor(sqr_size=5, latent_dim=NOISE_DIM, latent_vec=latent_vec)
    model.fit(dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk])
    return model

def display_sample(generator_model, noise, alpha):
    images = image_with_eigenvectors(generator_model, noise, alpha)



if __name__ == '__main__':
    model = generate_model()
    dataset = load_dataset()
    dataset = normalize_dataset(dataset)
    model = train_model(model)
