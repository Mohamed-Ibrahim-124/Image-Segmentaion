import numpy as np 
from os import path 
from scipy.io import loadmat

def handle_mat_struct(matobject):
    SEGMENTATION = 'SEGMENTATION'
    BOUNDARIES = 'BOUNDARIES'
    for i in matobject['groundTruth'][0]:
        yield i[0]

