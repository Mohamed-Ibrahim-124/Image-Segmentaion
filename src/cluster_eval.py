from scipy.misc import imshow
import src.resource_reader as reader
from src.kmeans import kmeans, draw_clusters
from src.spectral_clustering import spectral_clustering, rbf, knn
from src.eval import fmeasure, conditional_entropy
import numpy as np
from os import path, makedirs, environ


def evaluate_kmeans(dir, k_clusters):
    dir = path.join(dir, str(k_clusters))
    # create directory if not exist
    if not path.exists(dir):
        makedirs(dir)
    for image, ground_truth, name in reader.request_data():
        assigns_file = path.join(dir, name) + '_' + str(image.shape[0]) + '.npy'
        if path.exists(assigns_file):
            assigns = np.load(assigns_file)
        else:
            _, assigns = kmeans(image.reshape(image.shape[0] * image.shape[1], image.shape[2]), k=k_clusters)
            np.save(assigns_file, assigns)
        # evaluate f measures and conditional entropies if not evaluated
        eval_file = path.join(dir, name) + '_' + str(image.shape[0]) + '_eval.npy'
        if not path.exists(eval_file):
            f_measures = []
            entropies = []
            for seg, _ in ground_truth:
                seg = seg.flatten()
                f_measures.append(fmeasure(assigns, seg))
                entropies.append(conditional_entropy(assigns, seg))
            np.savetxt(eval_file, np.vstack((np.asmatrix(f_measures), np.asmatrix(entropies))))


def evaluate_spectral_clustering(dir, k_clusters, sim_func, sim_arg):
    dir = path.join(dir, str(k_clusters), str(sim_func).split()[1])
    if not path.exists(dir):
        makedirs(dir)
    for image, ground_truth, name in reader.request_data():
        assigns_file = path.join(dir, name) + '_' + str(image.shape[0]) + '.npy'
        if path.exists(assigns_file):
            assigns = np.load(assigns_file)
        else:
            _, assigns = spectral_clustering(image.reshape(image.shape[0] * image.shape[1], image.shape[2]), k=k_clusters, sim_func=sim_func, sim_arg=sim_arg)
            np.save(assigns_file, assigns)
        eval_file = path.join(dir, name) + '_' + str(image.shape[0]) + '_eval.npy'
        if not path.exists(eval_file):
            f_measures = []
            entropies = []
            for seg, _ in ground_truth:
                seg = seg.flatten()
                f_measures.append(fmeasure(assigns, seg))
                entropies.append(conditional_entropy(assigns, seg))
            np.savetxt(eval_file, np.vstack((np.asmatrix(f_measures), np.asmatrix(entropies))))


def load_eval_data(path):
    temp = np.loadtxt(path)
    # return f_measures, entropies
    return temp[0, :], temp[1, :]


if __name__ == '__main__':
    KMEANS_DIR = "../kmeans_eval"
    SPECTRAL_DIR = "../spectral_eval"
    environ['MKL_DYNAMIC'] = 'false'
    for k in [3, 5, 7, 9, 11]:
        evaluate_kmeans(KMEANS_DIR, k)
        for gamma in [1, 10]:
            evaluate_spectral_clustering(SPECTRAL_DIR, k, rbf, gamma)
        for n_neighbours in [3, 5]:
            evaluate_spectral_clustering(SPECTRAL_DIR, k, knn, n_neighbours)