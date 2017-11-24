from os import path, getcwd, listdir
import numpy as np
from scipy.misc import imread, imresize
from scipy.io import loadmat
from misc import handle_mat_struct
#downresizing to accelerate computation 
RES = (500, 500)

DATASET_PATH = path.join(
    path.split(getcwd())[0],
    "BSR",
    "BSDS500",
    "data",
    "images",
    "test"
)

GROUND_TRUTH = path.join(
    path.split(getcwd())[0],
    "BSR",
    "BSDS500",
    "data",
    "groundTruth",
    "test"
)



def _load_testset(num=500):
    abs_file_path = sorted(
        map(lambda x: path.join(DATASET_PATH, x), listdir(DATASET_PATH))
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


def _load_groundtruths(num=500):
    abs_file_path = map(
        lambda x: path.join(GROUND_TRUTH, x),
        listdir(GROUND_TRUTH)
    )
    p = sorted(abs_file_path)
    for fd in p :
        try:
            #load matlib file
            x = loadmat(fd)
            yield ( handle_mat_struct(x), fd)
        except OSError and ValueError as identifier:
            #pass errors for non image files 
            pass

def request_data():
    for img,truth in zip(_load_testset(), _load_groundtruths()):
        assert getfname(img[1]) == getfname(truth[1])
        yield img[0], truth[0] #only return files indx = 0 pass on filename

# test that loaded goundtruth has its corresponding  test file loaded correctly 
# for (i,i_fd), (j, j_fd) in zip(_load_testset(), _load_groundtruths()):
#     # print(i, j
#     # print("1")
#     getfname = lambda x: os.path.split(x)[1].split(".")[0]
#     assert getfname(i_fd) == getfname(j_fd)
getfname = lambda x: path.split(x)[1].split(".")[0]
if __name__ == "__main__":
    import inspect 
    
    img, gt_iter = next(request_data())
    gt_e = next(gt_iter)
    print(gt_e['Boundaries'][0])
