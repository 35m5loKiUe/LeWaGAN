import numpy as np
from scipy import linalg as LA
import tensorflow as tf
<<<<<<< HEAD
import time
=======

>>>>>>> 7f395c16cc1fc8687f9b95daa7d43b10072a7df9
from LeWaGAN.interface.params import NOISE_DIM

def generate_noise(number_of_noise_vectors):
    return tf.random.normal(shape=(number_of_noise_vectors, NOISE_DIM))


<<<<<<< HEAD
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
=======
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
>>>>>>> 7f395c16cc1fc8687f9b95daa7d43b10072a7df9

    return np.array(vectors)


<<<<<<< HEAD
def image_with_eigenvectors(model, noise, alpha, eigenvectors) :
=======
def image_with_eigenvectors(generator_model, noise, alpha) :
>>>>>>> 7f395c16cc1fc8687f9b95daa7d43b10072a7df9
    """This function generates an image with control over the noise (with eigenvecotrs)
    Args :
    alpha ; 1 by default, scalar to apply same weight to each eigenvector, array of size k to apply
    different weight to each vectors.
    """
    assert isinstance(alpha, (list, np.array)) == True

<<<<<<< HEAD
    #generate updated noise
    c = 0
    for vector in eigenvectors :
=======
    #Get the weight matrix and compute A*A_T
    weight_matrix = generator_model.trainable_variables[0].numpy()

    #Compute eigenvectors (list of k vectors, callable by index)
    vectors = eigenvectors(weight_matrix, k=10)

    #generate a new noise
    c = 0
    for vector in vectors :
>>>>>>> 7f395c16cc1fc8687f9b95daa7d43b10072a7df9
        noise += vector*alpha[c]
        c += 1

    final_noise = noise

    #generate an image
<<<<<<< HEAD
    generated_images = np.squeeze(model.generator(final_noise))*255
    return generated_images.astype(np.uint8)
=======
    generated_images = generator_model(final_noise)
    generated_images = np.squeeze(generated_images(noise))
    img = np.squeeze(generated_images(noise))*255

    return img.astype(np.uint8)
>>>>>>> 7f395c16cc1fc8687f9b95daa7d43b10072a7df9
