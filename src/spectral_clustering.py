import numpy as np
import os
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from sklearn.metrics.pairwise import rbf_kernel
from src.kmeans import kmeans, draw_clusters


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
    return np.divide(data, np.linalg.norm(data, axis=1)[:, None])


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
    # Generating similarity matrix
    if sim_mat is None:
        sim_mat = sim_func(data, sim_arg)
        # sim_mat = np.asmatrix(np.load('rbf_sim_mat.npy'))
    # Computing laplacian matrix
    laplace_matrix = laplacian(sim_mat, normed=False)
    # computing eigen values and  vectors
    eig_values, eig_vectors = np.linalg.eigh(laplace_matrix)
    # eig_vectors = np.asmatrix(np.load('eig_vec.npy'))
    # Normalizing each row
    normalized_data = normalize_rows(eig_vectors[:, :k])
    return kmeans(np.ma.array(normalized_data, mask=np.isnan(normalized_data)), 5, 0.0001, k=k)


def rbf(data, gamma):
    return rbf_kernel(data, None, gamma)


if __name__ == '__main__':
    os.environ['MKL_DYNAMIC'] = 'false'
    from scipy.misc import imread, imshow, imresize
    train_image = imresize(imread('../BSR/bench/data/images/8068.jpg'), (100, 100))
    image_shape = train_image.shape
    train_image = train_image.reshape((train_image.shape[0] * train_image.shape[1], train_image.shape[2]))
    k_clusters = 5
    centers, assigns= spectral_clustering(np.asmatrix(train_image), k_clusters, rbf, 10)
    image2 = draw_clusters(assigns, k_clusters, image_shape)
    imshow(image2)
