import numpy as np

import utils

if __name__ == '__main__':
    X_norm, _ = utils.create_data()
    A = utils.create_affinity_matrix(X_norm)
    
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigvals, eigvecs = np.linalg.eig(L)
    
    n_dim = eigvecs.shape[0]
    p = np.zeros(n_dim)
    p[eigvecs[:,1] > 0] = 1.0
    
    utils.show_result(X_norm, p)
