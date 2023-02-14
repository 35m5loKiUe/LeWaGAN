import numpy as np
from scipy import linalg as LA

def get_eigenvectors(matrix, k=10) :
    """This function computes the k most important eigenvectors from
    a matrix of weight"""

    #Compute AT*A
    A = np.dot(matrix, matrix.transpose())

    #Compute eigenvalues and eigenvectors
    w, v = LA.eig(A)

    #Get the index of k most important eigenvalues
    sort = sorted(range(len(w)), key=lambda k: w[k], reverse=True)[:k]

    #Get the k most important eigenvectors
    vectors = [v[i] for i in sort ]

    return np.array(vectors)
