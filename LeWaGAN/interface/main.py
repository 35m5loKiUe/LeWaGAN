from LeWaGAN.interface.params import NOISE_DIM, BATCH_SIZE, EPOCHS, IMAGE_SIZE, NB_FILTERS
from LeWaGAN.model.classes import WGAN, GANMonitor
from LeWaGAN.model.registery import save_model, save_eigenvectors, load_eigenvectors, load_model
from LeWaGAN.data.dataset import load_dataset, normalize_dataset
from LeWaGAN.postprocessing.generate import image_with_eigenvectors, generate_noise, eigenvectors
import matplotlib.pyplot as plt

def generate_model():
    model = WGAN(
        image_size=IMAGE_SIZE,
        nb_filters=NB_FILTERS,
        latent_dim=NOISE_DIM,
        discriminator_extra_steps=3,
        gp_weight=10.0
        )
    model.compile()
    return model

def train_model(model, dataset):
    latent_vec=generate_noise(5**2)
    cbk = GANMonitor(sqr_size=5, latent_dim=NOISE_DIM, latent_vec=latent_vec)
    model.fit(dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk])
    return model


if __name__ == '__main__':
    model = generate_model()
    #dataset = load_dataset()
    #dataset = normalize_dataset(dataset)
    #model = train_model(model, dataset)
    #model = save_model(model)
    #eig = eigenvectors(model, 5)
    #eig = save_eigenvectors(eig)
    model = load_model(model)
    eig = load_eigenvectors()
    noise = generate_noise(1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = image_with_eigenvectors(
            model=model,
            noise=noise,
            alpha = [0,0,0,0,0],
            eigenvectors=eig
            )
        plt.imshow(img)
        plt.axis('off')
    plt.show()
