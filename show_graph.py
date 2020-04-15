import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    X_norm, _ = utils.create_data()
    A = utils.create_affinity_matrix(X_norm)
    
    nb_data = len(X_norm)
    nodes = np.arange(nb_data)

    pos = {i: x for i, x in zip(nodes, X_norm)}
    edges = []
    for i, e in zip(nodes, A):
        adjacent = nodes[np.where(e==1)]
        for a in adjacent:
            edges.append((i, a))
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=30, edge_color='orange')
    plt.show()
