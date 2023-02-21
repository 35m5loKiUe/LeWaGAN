import numpy as np
from scipy import linalg as LA
import tensorflow as tf

from LeWaGAN.interface.params import NOISE_DIM

def generate_noise(number_of_noise_vectors):
    return tf.random.normal(shape=(number_of_noise_vectors, NOISE_DIM))


def eigenvectors(matrix, k=10) :
    """This function computes the k most important eigenvectors from
    a matrix of weight
    it returns a list of k vectors callable by index"""

    #Compute AT*A
    A = np.dot(matrix, matrix.transpose())

    #Compute eigenvalues and eigenvectors
    w, v = LA.eig(A)

    #Get the index of k most important eigenvalues
    sort = sorted(range(len(w)), key=lambda k: w[k], reverse=True)[:k]

    #Get the k most important eigenvectors
    vectors = [v[i] for i in sort ]

    return np.array(vectors)


def image_with_eigenvectors(generator_model, noise, alpha) :
    """This function generates an image with control over the noise (with eigenvecotrs)
    Args :
    alpha ; 1 by default, scalar to apply same weight to each eigenvector, array of size k to apply
    different weight to each vectors.
    """
    assert isinstance(alpha, (list, np.array)) == True

    #Get the weight matrix and compute A*A_T
    weight_matrix = generator_model.trainable_variables[0].numpy()

    #Compute eigenvectors (list of k vectors, callable by index)
    vectors = eigenvectors(weight_matrix, k=10)

    #generate a new noise
    c = 0
    for vector in vectors :
        noise += vector*alpha[c]
        c += 1

    final_noise = noise

    #generate an image
    generated_images = generator_model(final_noise)
    generated_images = np.squeeze(generated_images(noise))
    img = np.squeeze(generated_images(noise))*255
    return img.astype(np.uint8)
