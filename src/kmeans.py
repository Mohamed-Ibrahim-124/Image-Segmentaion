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
    :return: new centroids
    :rtype: numpy matrix, shape(n_clusters, n_features)
    """
    # Sorting data samples such that samples belong to the same cluster are grouped together.
    order = np.argsort(assignments)
    data = data[order]
    # calculating number of samples per cluster
    clusters, spc = np.unique(assignments, return_counts=True)
    # computing new centroids
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


def draw_clusters(assignments, k, shape):
    """
    draw data samples as image of colored clusters

    :param assignments: define assigned cluster per each sample
    :type assignments: numpy matrix, shape(n_samples, 1)
    :param k: number of clusters
    ;:type k: int
    :param shape: define the dimension of image needed to represent colored samples
    ;:type shape: tuple(n_rows, m_columns)
    :return:
    """
    colors = [[i, j, l] for i, j, l in zip(sample(range(0, 255, 5), k), sample(range(0, 255, 5), k),
                                           sample(range(0, 255, 5), k))]
    image = np.zeros((shape[0], shape[1], shape[2]))
    for m in range(shape[0]):
        for n in range(shape[1]):
            image[m, n] = colors[assignments[m * shape[1] + n]]
    return image


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
    sum_err = None
    for i in range(max_iters):
        # calculate distances
        distances = distance_func(data, centroids)
        # Assigning samples to clusters
        assignments = np.argmin(distances, axis=1)
        # Computing new centroids
        new_centroids = compute_new_centroids(data, assignments)
        # computing error
        sum_err = compute_error(new_centroids, centroids)
        if sum_err < threshold:
            return new_centroids, assignments, sum_err
        centroids = new_centroids
    return centroids, assignments, sum_err


if __name__ == '__main__':
    from scipy.misc import imread, imsave
    train_image = imread('BSR/bench/data/images/8068.jpg')
    image_shape = train_image.shape
    train_image = train_image.reshape((train_image.shape[0] * train_image.shape[1], 3))
    k_clusters = 5
    centers, assigns, error = kmeans(np.asmatrix(train_image), 0.00001, k_clusters)
    image2 = draw_clusters(assigns, k_clusters, image_shape)
    imsave('test.png', image2)
