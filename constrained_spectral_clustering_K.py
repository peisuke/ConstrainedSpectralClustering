import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import utils

if __name__ == '__main__':
    K = 4
    X_norm, z = utils.create_data()
    X_norm = np.concatenate((X_norm, X_norm + (0, 3.2)))
    z = np.concatenate((z, z + 2))

    A = utils.create_affinity_matrix(X_norm)
    Q = utils.create_constraint_matrix(z)
    
    D = np.diag(np.sum(A, axis=1))
    vol = np.sum(A)

    D_norm = np.linalg.inv(np.sqrt(D))
    L_norm = np.eye(*A.shape) - D_norm.dot(A.dot(D_norm))
    Q_norm = D_norm.dot(Q.dot(D_norm))

    # alpha < K-th eigenval of Q_norm
    alpha = 0.6 * sp.linalg.svdvals(Q_norm)[K]
    Q1 = Q_norm - alpha * np.eye(*Q_norm.shape)
    
    val, vec = sp.linalg.eig(L_norm, Q1)
    
    vec = vec[:,val >= 0]
    vec_norm = (vec / np.linalg.norm(vec, axis=0)) * np.sqrt(vol)

    costs = np.multiply(vec_norm.T.dot(L_norm), vec_norm.T).sum(axis=1)
    ids = np.where(costs > 1e-10)[0]
    min_idx = np.argsort(costs[ids])[0:K]
    min_v = vec_norm[:,ids[min_idx]]

    u = D_norm.dot(min_v)
   
    model = KMeans(n_clusters=K).fit(u)
    labels = model.labels_
    
    utils.show_result(X_norm, labels)
