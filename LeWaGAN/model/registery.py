from LeWaGAN.Interface.params import ROOT_PATH, MODEL_PATH
import os
import pickle
import joblib


# Sauvegerde le preprocessing en local ou sur GCP
#def save_preprocessing(where=PREPROC_LOCATION, preproc_data):
#    if == 'LOCAL':
#        with open(os.path.join(ROOT_PATH, PREPROC_PATH, f'dataset_{IMAGE_SIZE}'),'wb') as file :
#            pickle.dump(preproc_data, file)
#    return preproc_data

# Sauvegerde le preprocessing en local ou sur GCP
#def load_preprocessing(where=PREPROC_LOCATION):
#    if == 'LOCAL':
#        with open(os.path.join(ROOT_PATH, PREPROC_PATH, f'dataset_{IMAGE_SIZE}'),'wb') as file :
#            preproc_data = pickle.load(preproc_data, file)
#    return preproc_data

def load_model():
    model_path = os.path.join(ROOT_PATH, MODEL_PATH)
    return joblib.load(model_path+'/model.pkl')
