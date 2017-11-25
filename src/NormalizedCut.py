import numpy as np
import scipy.sparse.csgraph.laplacian as lp
from sklearn.metrics.pairwise import rbf_kernel as rbf
import sklearn.neighbors.kneighbors_graph as kneighbors_graph
from numpy.linalg import eigh

from src import kmeans

def normalize_rows(data):
    return np.divide(data, np.linalg.norm(data, axis=1)[:, None])

def rbf_sim(data,gama):
    # may need resize of shape of data in case of less than 4GB of RAMs
    sim_mat=rbf(data, None, gama)
    return sim_mat

def NN_sim(data,N):
   return kneighbors_graph(data, N, mode='connectivity', include_self=True).toarray()

def Norm_cut(data,simfunc=NN_sim,sim_parm=5,k=5):
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
    #Compute similarity matrix
    temp=simfunc(data,sim_parm)
    temp=lp(temp)
    # get eigen value , vector
    eigen_value,eigenVector=eigh(temp)
    #get smallest eigen value ,vector
    temp=eigenVector[:,k]
    #normalize the rows
    temp=normalize_rows(temp)
    #compute kmeans
    numrepeated=5
    temp = kmeans.evaluate_kmeans(train_image, numrepeated, 0.0001, k)
    return temp

if __name__ == '__main__':
    from scipy.misc import imshow
    from src import resource_reader as rr
    train_image = next(rr.request_data())[0]
    image_shape = train_image.shape
    train_image = train_image.reshape((train_image.shape[0] * train_image.shape[1], 3))
    k_clusters = 5
    best = Norm_cut(train_image, 5, 0.0001, k_clusters)
