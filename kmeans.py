from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from random import sample


def gen_centroids(data, k):
    """
    Choose random k samples as initial centroids

    :param data: data samples
    :type data: numpy matrix, shape(n_samples, n_features)
    :param k: number of clusters (centroids)
    :type k: int
    :return: centroids matrix
    :rtype: numpy matrix, shape(k, n_features)
    """
    centroids = np.zeros((k, data.shape[1]))
    for i, x in enumerate(sample(range(data.shape[0]), k)):
        centroids[i] = data[x]
    return np.asmatrix(centroids)


def compute_new_centroids(data, assignments):
    """
    Compute the new centroids based on data samples assignments

    :param data: data samples.
    :type data: numpy matrix, shape(n_samples, n_features)
    :param assignments: determine cluster for each samples.
    :type assignments: numpy matrix, shape(n_samples, 1)
    :param k: number of clusters (centroids)
    :type k: int
    :return: new centroids
    :rtype: numpy matrix, shape(n_clusters, n_features)
    """
    order = np.argsort(assignments)
    data = data[order]
    clusters, spc = np.unique(assignments, return_counts=True)
    centroids = np.zeros((len(clusters), data.shape[1]))
    start = 0
    for i, x in enumerate(spc):
        centroids[i] = np.mean(data[start: start + x], axis=0)
        start += x
    return np.asmatrix(centroids)


def compute_error(new_centroids, centroids):
    """
    Compute Error ratio between centroids

    :param new_centroids: new centroids
    :type new_centroids: numpy matrix, shape(n_clusters, n_features)
    :param centroids: previous centroids
    :type centroids: numpy matrix, shape(n_clusters, n_features)
    :return: error ratio between new and old centroids
    :rtype: float
    """
    sum_err = 0
    for x, y in zip(new_centroids, centroids):
        sum_err += np.linalg.norm(x - y)
    return sum_err


def kmeans(data, threshold, k=2, max_iters=100, distance_func=euclidean_distances, centroids=None):
    """
    Compute k-means clustering on given data samples

    :param data: data samples.
    :type data: numpy matrix, shape(n_samples, n_features)
    :param threshold: max error ratio allowed.
    :type threshold: float, (optional)
    :param k: number of clusters
    :type k: int
    :param max_iters: maximum number of iterations
    :type max_iters: int, (optional)
    :param distance_func: function used to compute the
    :param centroids: k initial centroids
    :type centroids: numpy matrix, (optional), shape(k_samples, n_features)
    :return: centroids, samples assignments
    :rtype: numpy matrix, shape(n_clusters, n_features), numpy matrix, shape(n_samples, 1)
    """
    if centroids is None:
        # selecting Random centroids from the given data
        centroids = gen_centroids(data, k)
    else:
        if centroids.shape != (k, data.shape[1]):
            raise Exception("Error, Number of centroids given is not the same as number of clusters specified.")
    assignments = None
    for i in range(max_iters):
        # Calculate Distances
        distances = distance_func(data, centroids)
        # Assigning samples to clusters
        assignments = np.argmin(distances, axis=1)
        # Computing new centroids
        new_centroids = compute_new_centroids(data, assignments)
        # computing error
        sum_err = compute_error(new_centroids, centroids)
        if sum_err < threshold:
            return new_centroids, assignments
        centroids = new_centroids
    return centroids, assignments


if __name__ == '__main__':
    from scipy.misc import imread
    image = imread('BSR/bench/data/images/2018.jpg')
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans(np.asmatrix(image), 0.00001, 4)
