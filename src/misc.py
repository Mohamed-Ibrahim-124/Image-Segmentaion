import numpy as np 
from os import path 
from scipy.io import loadmat
from scipy.misc import imresize
import logging

# logging.basicConfig("misc.py", level=logging.DEBUG)
def handle_mat_struct(matobject):
    from resource_reader import RES
    SEGMENTATION = 'Segmentation'
    BOUNDARIES = 'Boundaries'

    for i in matobject['groundTruth'][0]:
        yield (
            imresize(i[0][SEGMENTATION][0], RES),
            imresize(i[0][BOUNDARIES][0], RES)
            )

def construct_knn_graph(adj_matrix, n=5):
    for row, row_num in zip(adj_matrix, range(adj_matrix.shape[0])):
        max_indx_n = np.argsort(row)[-n:]
        assert len(max_indx_n) == n
        assert len(row) > 1
        for i in range(len(row)):
            adj_matrix[row_num, i] = 0 if i not in max_indx_n else 1
    return adj_matrix

def construct_knn_graph_spatial_layout(sim_matrix, spatial_nearest_neighbours=24, n=5):
    from scipy.spatial import KDTree
    from resource_reader import RES
    x,y = np.mgrid[0:RES[0], 0:RES[1]]
    tree = KDTree(list(zip(
        x.ravel(),
        y.ravel()
    )))
    assert tree.data.shape[0] == RES[0] * RES[1]
    _, knn_indices = tree.query(tree.data, k=spatial_nearest_neighbours)
    for row, row_num in zip(sim_matrix, range(sim_matrix.shape[0])):
        for i in range(len(row)):
            if i not in knn_indices[row_num] : sim_matrix[row_num, i] = 0
    return construct_knn_graph(sim_matrix, n=5)

def compute_degree_matrix(adj_matrix):
    degree_matrix = np.zeros((adj_matrix.shape))
    for i,row in zip(range(adj_matrix.shape[0]), adj_matrix):
        degree_matrix[i,i] = sum(row)
    return degree_matrix


def add_spatial(data):
    spatial_data = np.zeros((data.shape[0], data.shape[1], data.shape[2] + 2))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            spatial_data[i, j] = np.hstack((data[i, j], np.asarray([i, j])))
    return spatial_data


if __name__ =="__main__":
# if True:
    #TODO test adjacency and knn graph construction 
    rbf = lambda x,y,g : np.exp(-1 * np.abs(x-y)**2 / (2 * g**2))
    from resource_reader import request_data
    from sklearn.metrics.pairwise import rbf_kernel
    data = next(request_data())[0]
    data = data.reshape(data.shape[0] * data.shape[1],3)
    print(data.shape)
    print(
        # construct_adjacency_matrix(data, rbf, 1)
        rbf_kernel(data, gamma=1).shape
    )
    print(
        construct_knn_graph_spatial_layout(rbf_kernel(data))
    )