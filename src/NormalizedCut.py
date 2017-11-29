from sklearn.metrics.pairwise import rbf_kernel as rbf
from numpy.linalg import eigh
from src import kmeans
from scipy.sparse.csgraph import laplacian as lp
from sklearn.neighbors import kneighbors_graph
import numpy as numpy
import matplotlib.pyplot as plt
from scipy.misc import *
from src import resource_reader as rr


def normalize_rows(data):
    return numpy.divide(data, numpy.linalg.norm(data, axis=1)[:, None])


def rbf_sim(data, gamma):
    sim_mat = rbf(data, None, gamma)
    return sim_mat


def nn_sim(data, n):
    return kneighbors_graph(data, n, mode='connectivity', include_self=True).toarray()


def Norm_cut(data, simfunc=rbf_sim, sim_parm=10, k=5):
    """
    Compute Normalized-cut clustering on given data samples
    :param data: data samples.
    :type data: numpy matrix, shape(n_samples, n_features)
    :param simfunc: Similarity function used in calculating Similarity Matrix
    :type simfunc: Function, (optional)
    :param sim_parm: Gama parameter used im RBF Kernel or N-Nearest Neighbour
    :type sim_parm: float
    :param k: number of clusters
    :type k: int
    :return: centroids, samples assignments
    :rtype: numpy matrix, shape(n_clusters, n_features), numpy matrix, shape(n_samples, 1)
    """
    # Compute similarity matrix
    temp = simfunc(data, sim_parm)
    temp = lp(temp, return_diag=False)
    # get eigen value , vector
    temp = numpy.reshape(temp, (10000, 10000))
    eigen_value, eigenVector = eigh(temp)
    # get smallest eigen value ,vector
    temp = eigenVector[:, k]
    # normalize the rows
    temp = normalize_rows(numpy.matrix(temp))
    # compute kmeans
    numrepeated = 5
    temp = kmeans.evaluate_kmeans(train_image, numrepeated, 0.0001, k)
    return temp


if __name__ == '__main__':
    train_image = next(rr.request_data())[0]
    train_image = imresize(train_image, (100, 100))

    image_shape = train_image.shape
    train_image = train_image.reshape((train_image.shape[0] * train_image.shape[1], 3))
    # number of clusters {3,5,7,9,11}
    k_clusters = 5
    # gamma value {1,10} or number of Neighbours 5NN in our case
    simparm = 10
    best = Norm_cut(data=train_image, simfunc=nn_sim, sim_parm=10,k=k_clusters)
    image2 = kmeans.draw_clusters(best[1], k_clusters, image_shape, best[0])
    plt.imshow(image2)
    plt.show()
