import numpy as np 
from os import path 
from scipy.io import loadmat
from scipy.misc import imresize
import logging
# logging.basicConfig("misc.py", level=logging.DEBUG)
def handle_mat_struct(matobject):
    SEGMENTATION = 'Segmentation'
    BOUNDARIES = 'Boundaries'

    for i in matobject['groundTruth'][0]:
        yield (
            i[0][SEGMENTATION][0],
            i[0][BOUNDARIES][0]
            )

def construct_knn_graph(adj_matrix, n=5):
    for row, row_num in zip(adj_matrix, range(adj_matrix.shape[0])):
        max_indx_n = np.argsort(row)[:n]
        assert len(max_indx_n) == n
        assert len(row) > 1
        for i in range(len(row)):
            adj_matrix[row_num, i] = 0 if i not in max_indx_n else 1
    return adj_matrix

if __name__ =="__main__":
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
        construct_knn_graph(rbf_kernel(data, gamma=1))
    )