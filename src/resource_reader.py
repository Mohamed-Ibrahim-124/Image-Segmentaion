from os import path, getcwd, listdir
from scipy.misc import imread, imresize
from scipy.io import loadmat
from misc import handle_mat_struct

# down resizing to accelerate computation
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


def _load_test_set():
    abs_file_path = sorted(
        map(lambda z: path.join(DATASET_PATH, z), listdir(DATASET_PATH))
    )
    for fd in abs_file_path:
        try:
            x = imresize(
                imread(fd),
                RES
            )
            yield x, fd
        except OSError:
            # pass errors for reading non image files
            pass


def _load_ground_truths():
    abs_file_path = map(
        lambda z: path.join(GROUND_TRUTH, z),
        listdir(GROUND_TRUTH)
    )
    p = sorted(abs_file_path)
    for fd in p:
        try:
            # load mat lib file
            x = loadmat(fd)
            yield (handle_mat_struct(x), fd)
        except OSError and ValueError:
            # pass errors for non image files
            pass


def request_data():
    for img, truth in zip(_load_test_set(), _load_ground_truths()):
        assert get_fname(img[1]) == get_fname(truth[1])
        yield img[0], truth[0]  # only return files index = 0 pass on filename


# test that loaded ground truth has its corresponding  test file loaded correctly
# for (i,i_fd), (j, j_fd) in zip(_load_test_set(), _load_ground_truths()):
#     # print(i, j
#     # print("1")
#     get_fname = lambda x: os.path.split(x)[1].split(".")[0]
#     assert get_fname(i_fd) == get_fname(j_fd)


get_fname = lambda x: path.split(x)[1].split(".")[0]

if __name__ == "__main__":
    image, gt_iter = next(request_data())
    gt_e = next(gt_iter)
    print(gt_e[0])
