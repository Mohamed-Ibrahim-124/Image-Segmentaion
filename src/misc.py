import numpy as np 
from os import path 
from scipy.io import loadmat
from scipy.misc import imresize

def handle_mat_struct(matobject):
    SEGMENTATION = 'Segmentation'
    BOUNDARIES = 'Boundaries'
    
    for i in matobject['groundTruth'][0]:
        yield (
            i[0][SEGMENTATION][0],
            i[0][BOUNDARIES][0]
            )

