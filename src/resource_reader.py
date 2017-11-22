import os 
import numpy as np
from scipy.misc import imread, imresize
from scipy.io import loadmat

#downresizing to accelerate computation 
RES = (30, 30)

DATASET_PATH = os.path.join(
    os.path.split(os.getcwd())[0],
    "BSR",
    "BSDS500",
    "data",
    "images",
    "test"
)

GROUND_TRUTH = os.path.join(
    os.path.split(os.getcwd())[0],
    "BSR",
    "BSDS500",
    "data",
    "groundTruth",
    "test"
)



def load_testset(num=500):
    abs_file_path = sorted(
        map(lambda x: os.path.join(DATASET_PATH, x), os.listdir(DATASET_PATH))
    )
    for fd in abs_file_path:
        try:
            x =  imresize(
                imread(fd),
                RES
            )
            yield x,fd
        except OSError as err:
            #pass errors for reading nonimage files
            pass


def load_groundtruths(num=500):
    abs_file_path = map(
        lambda x: os.path.join(GROUND_TRUTH, x),
        os.listdir(GROUND_TRUTH)
    )
    p = sorted(abs_file_path)
    for fd in p :
        try:
            #load matlib file
            x = loadmat(fd)
            yield x, fd
        except OSError as identifier:
            #pass errors for non image files 
            pass

def request_data():
    for img,truth in zip(load_testset(), load_groundtruths()):
        yield img[0], truth[0]

# test that loaded goundtruth has its corresponding  test file loaded correctly 
for (i,i_fd), (j, j_fd) in zip(load_testset(), load_groundtruths()):
    # print(i, j)
    # print("1")
    getfname = lambda x: os.path.split(x)[1].split(".")[0]
    assert getfname(i_fd) == getfname(j_fd)