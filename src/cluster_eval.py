<<<<<<< HEAD
import resource_reader as reader
from kmeans import kmeans
from spectral_clustering import spectral_clustering, rbf, knn
from eval import fmeasure, conditional_entropy
=======
import src.resource_reader as reader
from src.kmeans import kmeans, draw_clusters
from src.spectral_clustering import spectral_clustering, rbf, knn
from src.eval import fmeasure, conditional_entropy
>>>>>>> bde5327d1297ce2558ae5b1520d7595fecf87902
import numpy as np
from os import path, makedirs, environ
from itertools import islice
from scipy.misc import imshow, imread, imresize

KMEANS_DIR = "../kmeans_eval"
SPECTRAL_DIR = "../spectral_eval"


def _evaluate_kmeans(dir, data, ground_truth, name, resolution, k_clusters, recompute=False):
    """
    Apply kmeans on given data then evaluate f-measure and conditional entropy for given ground truths.

    :param dir: path for directory to store output
    :type dir: str
    :param data: data samples
    :type data: np.matrix, shape(n_samples, n_features)
    :param ground_truth: iterator for ground truths obtained from resource reader
    :type ground_truth: iterator
    :param name: file name to store data output
    :type name: str
    :param resolution: image resolution
    :type resolution: int
    :param k_clusters: number of clusters
    :type k_clusters: int
    :param recompute: force compute assignments and evaluation files even if they already exist.
    :type recompute: bool
    """
    assert path.exists(dir), 'Given directory is not found'
    assigns_file = path.join(dir, name) + '_' + str(resolution) + '.npy'
    eval_file = path.join(dir, name) + '_' + str(resolution) + '.eval'
    if not path.exists(assigns_file) or not path.exists(eval_file) or recompute:
        # computing and saving assignments
        _, assigns = kmeans(data, k=k_clusters)
        del _, data
        np.save(assigns_file, assigns)
        # computing and saving evaluations
        f_measures = []
        entropies = []
        for seg, _ in ground_truth:
            seg = seg.flatten()
            f_measures.append(fmeasure(assigns, seg))
            entropies.append(conditional_entropy(assigns, seg))
        f_measures = np.asarray(f_measures)
        entropies = np.asarray(entropies)
        np.savetxt(eval_file, np.vstack((f_measures, entropies)))


def _evaluate_spectral(dir, data, ground_truth, name, resolution, k_clusters, sim_func, sim_arg, recompute=False):
    """
    Apply Spectral Clustering on given data then evaluate f-measure and conditional entropy for given ground truths.

    :param dir: path for directory to store output
    :type dir: str
    :param data: data samples
    :type data: np.matrix, shape(n_samples, n_features)
    :param ground_truth: iterator for ground truths obtained from resource reader
    :type ground_truth: iterator
    :param name: file name to store data output
    :type name: str
    :param resolution: image resolution
    :type resolution: int
    :param k_clusters: number of clusters
    :type k_clusters: int
    :param sim_func: can be rbf or knn
    :type sim_func: function
    :param sim_arg: gamma in case of rbf and n_neighbours in case of knn
    :type sim_arg: float for gamma, int for n_neighbours
    :param recompute: force compute assignments and evaluation files even if they already exist.
    :type recompute: bool
    """
    assert path.exists(dir), 'Given directory is not found'
    assigns_file = path.join(dir, name) + '_' + str(resolution) + '.npy'
    eval_file = path.join(dir, name) + '_' + str(resolution) + '.eval'
    if not path.exists(assigns_file) or not path.exists(eval_file) or recompute:
        # computing and saving assignments
        _, assigns = spectral_clustering(data, k=k_clusters, sim_func=sim_func, sim_arg=sim_arg)
        del _, data
        np.save(assigns_file, assigns)
        # computing and saving evaluations
        f_measures = []
        entropies = []
        for seg, _ in ground_truth:
            seg = seg.flatten()
            f_measures.append(fmeasure(assigns, seg))
            entropies.append(conditional_entropy(assigns, seg))
        f_measures = np.asarray(f_measures)
        entropies = np.asarray(entropies)
        np.savetxt(eval_file, np.vstack((f_measures, entropies)))


def evaluate_kmeans(dir, k_clusters, recompute=False):
    """
    Apply Spectral Clustering on given data then evaluate f-measure and conditional entropy for given ground truths.

    :param dir: path for directory to store output
    :type dir: str
    :param k_clusters: number of clusters
    :type k_clusters: int
    :param recompute: force compute assignments and evaluation files even if they already exist.
    :type recompute: bool
    """
    dir = path.join(dir, str(k_clusters))
    if not path.exists(dir):
        makedirs(dir)
    for image, ground_truth, name in islice(reader.request_data(), 100):
        _evaluate_kmeans(dir, image.reshape(image.shape[0] * image.shape[1], image.shape[2]), ground_truth, name,
                         image.shape[0], k_clusters, recompute)


def evaluate_spectral(dir, k_clusters, sim_func, sim_arg, recompute=False):
    """
    Apply Spectral Clustering on given data then evaluate f-measure and conditional entropy for given ground truths.

    :param dir: path for directory to store output
    :type dir: str
    :param k_clusters: number of clusters
    :type k_clusters: int
    :param sim_func: can be rbf or knn
    :type sim_func: function
    :param sim_arg: gamma in case of rbf and n_neighbours in case of knn
    :type sim_arg: float for gamma, int for n_neighbours
    :param recompute: force compute assignments and evaluation files even if they already exist.
    :type recompute: bool
    """
    dir = path.join(dir, str(k_clusters), str(sim_func).split()[1], str(sim_arg))
    if not path.exists(dir):
        makedirs(dir)
<<<<<<< HEAD
    for image, ground_truth, name in islice(reader.request_data(), 0,30):
=======
    for image, ground_truth, name in islice(reader.request_data(), 30):
>>>>>>> bde5327d1297ce2558ae5b1520d7595fecf87902
        _evaluate_spectral(dir, image.reshape(image.shape[0] * image.shape[1], image.shape[2]), ground_truth, name,
                           image.shape[0], k_clusters, sim_func, sim_arg, recompute)


def load_eval_data(path):
    """
    :param path: path to evaluation file
    :type path: str
    :return: (f_measure, conditional_entropies)
    :rtype: (nd-array, nd-array)
    """
    temp = np.loadtxt(path)
    # return f_measures, entropies
    return temp[0, :], temp[1, :]


def read_kmeans_eval(name, k, resolution):
    """
    :param name: image name
    :type name: str
    :param k: number of clusters
    :type k: int
    :param resolution: resolution of the image to load evaluation for.
    :type resolution: int
    :return: assignments, (f_measure, conditional_entropies)
    :rtype: nd-array, (nd-array, nd-array)
    """
    dir = path.join(KMEANS_DIR, str(k))
    name = str(name).split('.')[0] + '_' + str(resolution)
    assert path.exists(path.join(dir, name + '.npy')), 'Assignments file is missing or not found'
    assert path.exists(path.join(dir, name + '.eval')), 'Evaluation file is missing or not found'
    return np.load(path.join(dir, name + '.npy')), load_eval_data(path.join(dir, name + '.eval'))


def read_spectral_eval(name, k, resolution, sim_func, sim_arg):
    """
    :param name: image name
    :type name: str
    :param k: number of clusters
    :type k: int
    :param resolution: resolution of the image to load evaluation for.
    :type resolution: int
    :param sim_func: can be rbf or knn
    :type sim_func: function
    :param sim_arg: gamma in case of rbf and n_neighbours in case of knn
    :type sim_arg: float for gamma, int for n_neighbours
    :return: assignments, (f_measure, conditional_entropies)
    :rtype: nd-array, (nd-array, nd-array)
    """
    dir = path.join(SPECTRAL_DIR, str(k), str(sim_func).split()[1], str(sim_arg))
    name = str(name).split('.')[0] + '_' + str(resolution)
    assert path.exists(path.join(dir, name + '.npy')), 'Assignments file is missing or not found'
    assert path.exists(path.join(dir, name + '.eval')), 'Evaluation file is missing or not found'
    return np.load(path.join(dir, name + '.npy')), load_eval_data(path.join(dir, name + '.eval'))

def big_picture_eval():
    from visualize_data import visualize_data
    from itertools import islice
    from spectral_clustering import knn
    for img, gt_iter, fname in islice(reader.request_data(),5):
        kmeans_ouput = kmeans(img,k=5)
        visualize_data(kmeans_ouput, gt_iter, fname)
    for img, gt_iter, fname in islice(reader.request_data(),5):
        visualize_data(
            spectral_clustering(img,k=5),
            gt_iter,
            fname
        )
    for img, gt_iter,fname in islice(reader.request_data(), 5):
        visualize_data(
            kmeans(img, k=5),
            spectral_clustering(img, k=5, knn, 5)
        )

def avg_eval(path):
    f_measures, entropies = load_eval_data(path)
    return np.sum(f_measures) / f_measures.size, np.sum(entropies) / entropies.size


def avg_kmeans_eval(name, k, resolution):
    f_measures, entropies = read_kmeans_eval(name, k, resolution)[1]
    return np.sum(f_measures) / f_measures.size, np.sum(entropies) / entropies.size


def avg_spectral_eval(name, k, resolution, sim_func, sim_arg):
    f_measures, entropies = read_spectral_eval(name, k, resolution, sim_func, sim_arg)[1]
    return np.sum(f_measures) / f_measures.size, np.sum(entropies) / entropies.size


def total_avg_kmeans(name, resolution):
    f_sum = entropy_sum = f_length = entropy_length =0
    for k in [3, 5, 7, 9, 11]:
        f_measures, entropies = read_kmeans_eval(name, k, resolution)[1]
        f_sum += np.sum(f_measures)
        entropy_sum += np.sum(entropies)
        f_length += f_measures.size
        entropy_length += entropies.size
    return f_sum / f_length, entropy_sum / entropy_length


def total_avg_spectral(name, resolution, sim_func, sim_arg):
    f_sum = entropy_sum = f_length = entropy_length = 0
    for k in [3, 5, 7, 9, 11]:
        f_measures, entropies = read_spectral_eval(name, k, resolution, sim_func, sim_arg)[1]
        f_sum += np.sum(f_measures)
        entropy_sum += np.sum(entropies)
        f_length += f_measures.size
        entropy_length += entropies.size
    return f_sum / f_length, entropy_sum / entropy_length


if __name__ == '__main__':
    environ['MKL_DYNAMIC'] = 'false'
    for k in [3, 5, 7, 9, 11]:
        evaluate_kmeans(KMEANS_DIR, k)
        for gamma in [1, 10]:
            evaluate_spectral(SPECTRAL_DIR, k, rbf, gamma)
        for n_neighbours in [5]:
            evaluate_spectral(SPECTRAL_DIR, k, knn, n_neighbours)