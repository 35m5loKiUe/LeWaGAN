from LeWaGAN.interface.params import MODEL_LOCATION, ROOT_PATH, MODEL_PATH, IMAGE_SIZE
import os
import pickle



# Sauvegarde le model en local ou sur GCP
def save_model(model):
    if MODEL_LOCATION == 'LOCAL':
        with open(os.path.join(ROOT_PATH, MODEL_PATH, f'dataset_{IMAGE_SIZE}'),'wb') as file :
            pickle.dump(model, file)
    return model

# Sauvegarde le model en local ou sur GCP
def load_model():
    if MODEL_LOCATION == 'LOCAL':
        with open(os.path.join(ROOT_PATH, MODEL_PATH, f'dataset_{IMAGE_SIZE}'),'wb') as file :
            model = pickle.load(model, file)
    return model
