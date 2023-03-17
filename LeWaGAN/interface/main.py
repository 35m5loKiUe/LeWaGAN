from LeWaGAN.interface.params import NOISE_DIM, EPOCHS, IMAGE_SIZE, NB_FILTERS
from LeWaGAN.model.classes import WGAN, GANMonitor, GAN_ADA, WGAN_upsampling
from LeWaGAN.model.registery import save_model, save_eigenvectors, load_eigenvectors, load_model
from LeWaGAN.data.dataset import load_dataset, normalize_dataset
from LeWaGAN.postprocessing.generate import image_with_eigenvectors, generate_noise, eigenvectors
import matplotlib.pyplot as plt

def generate_model(model_type='gan_ada'):
    assert model_type in ['wgan', 'gan_ada', 'wgan_upsampling']
    if model_type == 'gan_ada':
        model = GAN_ADA(
            image_size=IMAGE_SIZE,
            nb_filters=NB_FILTERS,
            noise_dim=NOISE_DIM,
            dropout_rate=0.2,
            ema=0.99,
            leaky_relu_slope=0.2,
            initial_learning_rate = 2e-5,
            decay_steps=8000,
            decay_rate=0.5,
            beta_1=0.5
        )
    if model_type == 'wgan':
        model = WGAN(
            image_size=IMAGE_SIZE,
            nb_filters=NB_FILTERS,
            noise_dim=NOISE_DIM,
            discriminator_extra_steps=5,
            gp_weight=10.0,
            dropout_rate=0.2,
            leaky_relu_slope=0.2,
            initial_learning_rate = 2e-4,
            decay_steps=8000,
            beta_1=0.5
        )
    if model_type == 'wgan_upsampling':
        model = WGAN_upsampling(
            image_size=IMAGE_SIZE,
            nb_filters=NB_FILTERS,
            noise_dim=NOISE_DIM,
            discriminator_extra_steps=5,
            gp_weight=10.0,
            dropout_rate=0.2,
            leaky_relu_slope=0.2,
            initial_learning_rate = 2e-5,
            decay_steps=4729,
            beta_1=0.5
        )
    model.compile()
    return model

#def train_model(model, dataset):
#    latent_vec=generate_noise(5**2)
#    cbk = GANMonitor(sqr_size=5, latent_dim=NOISE_DIM, latent_vec=latent_vec)
#    model.fit(dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk])
#    return model

def train_model(model, dataset):
    latent_vec=generate_noise(5**2)
    cbk = GANMonitor(sqr_size=5, latent_dim=NOISE_DIM, latent_vec=latent_vec)
    model.fit(dataset, epochs=EPOCHS, callbacks=[cbk])
    return model


if __name__ == '__main__':
    model = generate_model(model_type='wgan')
    dataset = load_dataset()
    dataset = normalize_dataset(dataset)

    #model = save_model(model)
    model = load_model(model)
    model.compile()
    model = train_model(model, dataset)
    #eig = eigenvectors(model, 5)
    #eig = save_eigenvectors(eig)

    #eig = load_eigenvectors()
    #noise = generate_noise(1, 1)
    #
    #fig = plt.figure(figsize=(10, 10))
    #for i in range(16):
    #    plt.subplot(4, 4, i+1)
    #    img = image_with_eigenvectors(
    #        model=model,
    #        noise=noise,
    #        alpha = [i,0,0,0,0],
    #        eigenvectors=eig
    #        )
    #    plt.imshow(img)
    #    plt.axis('off')
    #plt.show()
