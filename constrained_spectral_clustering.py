import numpy as np
import scipy as sp
import utils

if __name__ == '__main__':
    X_norm, z = utils.create_data()
    A = utils.create_affinity_matrix(X_norm)
    Q = utils.create_constraint_matrix(z)
    
    D = np.diag(np.sum(A, axis=1))
    vol = np.sum(A)

    D_norm = np.linalg.inv(np.sqrt(D))
    L_norm = np.eye(*A.shape) - D_norm.dot(A.dot(D_norm))
    Q_norm = D_norm.dot(Q.dot(D_norm))

    # alpha < max eigenval of Q_norm
    alpha = 0.8 * sp.linalg.svdvals(Q_norm)[0]
    Q1 = Q_norm - alpha * np.eye(*Q_norm.shape)
    
    val, vec = sp.linalg.eig(L_norm, Q1)
    
    vec = vec[:,val >= 0]
    vec_norm = (vec / np.linalg.norm(vec, axis=0)) * np.sqrt(vol)

    costs = np.multiply(vec_norm.T.dot(L_norm), vec_norm.T).sum(axis=1)
    ids = np.where(costs > 1e-10)[0]
    min_idx = np.argmin(costs[ids])
    min_v = vec_norm[:,ids[min_idx]]

    u = D_norm.dot(min_v)
   
    n_dim = u.shape[0]
    p = np.zeros(n_dim)
    p[u > 0] = 1.0
    
    utils.show_result(X_norm, p)
