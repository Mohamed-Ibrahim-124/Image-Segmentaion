import numpy as np
import os
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from sklearn.metrics.pairwise import rbf_kernel
from src.kmeans import kmeans, draw_clusters
from sklearn.preprocessing import normalize as normalize
import matplotlib.pyplot as plt

def knn(data, neighbours):
    return kneighbors_graph(data, neighbours).todense()


def normalize_rows(data):
    """
    Normalize data matrix rows, by divide each row by it's norm.

    :param data: data samples
    :type data: numpy matrix, shape(n_samples, n_features)
    :return: normalized data matrix
    :rtype: numpy matrix, shape(n_samples, n_features)
    """
    return np.divide(data, normalize(data, norm='l1'))


def _spectral_clustering(data, sim_func, sim_arg=None, sim_mat=None):
    """
    Compute Normalized-cut clustering on given data samples

    :param data: data samples.
    :type data: numpy matrix, shape(n_samples, n_features)
    :param sim_func: Similarity function used in calculating Similarity Matrix, it can be RBF or KNN.
    :type sim_func: Function, (optional)
    :param sim_arg: Gama parameter used im RBF Kernel or number of neighbours in N-Nearest Neighbour
    :type sim_arg: float in RBF, int in KNN
    :param sim_mat: similarity matrix if already computed by user
    :type sim_mat: numpy matrix, shape(n_samples, n_samples)
    :return: eigen Vectors
    :rtype: numpy matrix, shape(n_clusters, n_clusters)
    """
    # Generating similarity matrix
    if sim_mat is None:
        sim_mat = sim_func(data, sim_arg)
    # Computing laplacian matrix
    laplace_matrix = laplacian(sim_mat, normed=False)
    del sim_mat
    # computing eigen vectors
    return np.linalg.eigh(laplace_matrix)[1]


def spectral_clustering(data, k, sim_func, sim_arg=None, sim_mat=None):
    """
    Compute Normalized-cut clustering on given data samples

    :param data: data samples.
    :type data: numpy matrix, shape(n_samples, n_features)
    :param sim_func: Similarity function used in calculating Similarity Matrix, it can be RBF or KNN.
    :type sim_func: Function, (optional)
    :param sim_arg: Gama parameter used im RBF Kernel or number of neighbours in N-Nearest Neighbour
    :type sim_arg: float in RBF, int in KNN
    :param k: number of clusters
    :type k: int
    :param sim_mat: similarity matrix if already computed by user
    :type sim_mat: numpy matrix, shape(n_samples, n_samples)
    :return: centroids, samples assignments
    :rtype: numpy matrix, shape(n_clusters, n_features), numpy matrix, shape(n_samples, 1)
    """
    normalized_data = normalize(_spectral_clustering(data, sim_func, sim_arg, sim_mat)[:, :k])
    return kmeans(normalized_data, 5, 0.0001, k=k)


def rbf(data, gamma):
    return rbf_kernel(data, None, gamma)


if __name__ == '__main__':
    # os.environ['MKL_DYNAMIC'] = 'false'
    from scipy.misc import imread, imshow, imresize
    from src.misc import construct_knn_graph_spatial_layout
    train_image = imresize(imread('../BSR/bench/data/images/8068.jpg'), (100, 100))
    image_shape = train_image.shape
    train_image = train_image.reshape((train_image.shape[0] * train_image.shape[1], train_image.shape[2]))
    k_clusters = 5
    rbf_sim=rbf(np.asmatrix(train_image),10)
    for i in[3,5,7,9,11]:
        sim=construct_knn_graph_spatial_layout((rbf_sim))
        centers, assigns= spectral_clustering(data=None, sim_mat=sim, k=i, sim_func=None, sim_arg=None)
        image2 = draw_clusters(assigns, k_clusters, image_shape)
        # imshow(image2)
        print(i)
        plt.imshow(image2)
        plt.show()
