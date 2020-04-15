import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from scipy import spatial
import matplotlib.pyplot as plt

def create_data():
    X, z = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)
    
    sc = preprocessing.StandardScaler()
    sc.fit(X)
    X_norm = sc.transform(X)
    return X_norm, z

def create_affinity_matrix(X):
    tree = spatial.KDTree(X)
    dist, idx = tree.query(X, k=16)
    
    idx = idx[:,1:]
    
    nb_data, _ = X.shape
    A = np.zeros((nb_data, nb_data))
    for i, j in zip(np.arange(nb_data), idx):
        A[i, j] = 1
    A = np.maximum(A.T, A)

    return A

def create_constraint_matrix(z):
    Q = 2 * (np.expand_dims(z, axis=1) == np.expand_dims(z, axis=0)).astype(int) - 1
    mask = (np.random.rand(*Q.shape) < 0.8).astype(int)
    mask = np.maximum(mask.T, mask)
    Q[mask == 1] = 0
    Q[np.arange(Q.shape[0]), np.arange(Q.shape[0])] = 1
    return Q

def show_result(X, z):
    plt.scatter(X[:,0], X[:,1], c=z)
    plt.show()
