import numpy as np
from scipy import linalg as LA
import tensorflow as tf
import time
from LeWaGAN.interface.params import NOISE_DIM
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def generate_noise(number_of_noise_vectors, seed=0):
    #If we only generate 1 vectors
    if number_of_noise_vectors == 1 :
        X = get_truncated_normal(mean=0, sd=1, low=-1, upp=1)
        noise = tf.convert_to_tensor(X.rvs(100, random_state=seed)) #keep the same random_state
        noise = tf.expand_dims(noise, axis=0)
        return noise



    noises = []
    #Generation of multiple noise vectors
    for k in range(number_of_noise_vectors) :
        X = get_truncated_normal(mean=0, sd=1, low=-1, upp=1)
        noise = tf.convert_to_tensor(X.rvs(100, random_state=seed+k)) #keep several random_states
        noises.append(noise)
    return tf.convert_to_tensor(noises)

def normalize_vector(v) :
    norm = LA.norm(v)
    if norm == 0 :
        return v
    return v / norm


def eigenvectors(model, k=10) :
    """This function computes the k most important eigenvectors from
    1st dense layer of generator
    it returns a list of k vectors callable by index"""

    #Get the weight matrix
    A = model.generator.layers[1].weights
    A = np.squeeze(A)
    start_time = time.time()
    print('\nComputing AT.A...')
    #Compute AT*A, remark AT and A are inverted in layers weights, so we compute A.AT instead
    ATA = np.dot(A, np.transpose(A))
    print(f'\nATA computed in {time.time() - start_time}s')
    start_time = time.time()
    print('\nComputing eigenvalues and eigenvectors...')
    #Compute eigenvalues and eigenvectors
    w, v = LA.eig(ATA)
    print(f'\neigenvalues and eigenvectors computed in {time.time() - start_time}s')
    #Get the index of k most important eigenvalues
    sort = sorted(range(len(w)), key=lambda k: w[k], reverse=True)[:k]
    #Get the k most important eigenvectors
    vectors = [normalize_vector(v[i]) for i in sort]

    return np.array(vectors)


def image_with_eigenvectors(model, noise, alpha, eigenvectors) :
    """This function generates an image with control over the noise (with eigenvecotrs)
    Args :
    alpha ; 1 by default, scalar to apply same weight to each eigenvector, array of size k to apply
    different weight to each vectors.
    """
    assert isinstance(alpha, (list, np.array)) == True

    #generate updated noise
    c = 0
    for vector in eigenvectors :
        noise += vector*alpha[c]
        c += 1

    final_noise = noise

    #generate an image
    generated_images = np.squeeze((model.generator(final_noise))+1)*255/2
    return generated_images.astype(np.uint8)
