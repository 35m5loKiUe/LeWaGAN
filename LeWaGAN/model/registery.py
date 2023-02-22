from LeWaGAN.interface.params import ROOT_PATH, MODEL_PATH, IMAGE_SIZE, NOISE_DIM, NB_FILTERS
import os
from tensorflow import keras
import pickle



# Sauvegarde le modele en local
def save_model(model):
    model.generator.save(os.path.join(ROOT_PATH, MODEL_PATH, f'generator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}'))
    print('\nModel saved in {}'.format(os.path.join(ROOT_PATH, MODEL_PATH, f'generator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}')))
    model.discriminator.save(os.path.join(ROOT_PATH, MODEL_PATH, f'discriminator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}'))
    print('\nModel saved in {}'.format(os.path.join(ROOT_PATH, MODEL_PATH, f'discriminator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}')))
    return model

# Charge le modele
def load_model(model):
    model.generator = keras.models.load_model(os.path.join(ROOT_PATH, MODEL_PATH, f'generator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}'))
    print('\nLoaded model {}'.format(os.path.join(ROOT_PATH, MODEL_PATH, f'generator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}')))
    model.discriminator = keras.models.load_model(os.path.join(ROOT_PATH, MODEL_PATH, f'discriminator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}'))
    print('\nLoaded model {}'.format(os.path.join(ROOT_PATH, MODEL_PATH, f'discriminator_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}')))
    return model

# Sauvegarde les vecteurs propres
def save_eigenvectors(eigenvectors):
    with open(os.path.join(ROOT_PATH, MODEL_PATH, f'eigenvectors_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}.pkl'),'wb') as file :
        pickle.dump(eigenvectors, file)
    print('\nEigenvectors saved in {}'.format(os.path.join(ROOT_PATH, MODEL_PATH, f'generator_{IMAGE_SIZE}.pkl')))
    return eigenvectors

# Charge les vecteurs propres
def load_eigenvectors():
    with open(os.path.join(ROOT_PATH, MODEL_PATH, f'eigenvectors_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}.pkl'),'rb') as file :
        eigenvectors = pickle.load(file)
    print('\nLoaded eigenvectors {}'.format(os.path.join(ROOT_PATH, MODEL_PATH, f'eigenvectors_IMSZ{IMAGE_SIZE}_NOIS{NOISE_DIM}_FILT{NB_FILTERS}.pkl')))
    return eigenvectors
