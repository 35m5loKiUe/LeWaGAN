import numpy as np
from scipy import linalg as LA
import tensorflow as tf
import time
from LeWaGAN.interface.params import NOISE_DIM

def generate_noise(number_of_noise_vectors):
    return tf.random.normal(shape=(number_of_noise_vectors, NOISE_DIM))


def eigenvectors(model, k=10) :
    """This function computes the k most important eigenvectors from
    1st dense layer of generator
    it returns a list of k vectors callable by index"""

    #Get the weight matrix
    A = model.generator.layers[0].weights
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
    vectors = [v[i] for i in sort]

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
    generated_images = np.squeeze(model.generator(final_noise))*255
    return generated_images.astype(np.uint8)
