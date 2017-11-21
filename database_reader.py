import os
from scipy.misc import imread
import numpy
import re


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load(directory='orl_faces', train_count=5):
    """
    This Function Load Images data set from a given directory into numpy.matrix\n
    Args:
    -----
    :param directory: Path to data set
    :type directory: String
    :param train_count: Number of images to load for training
    :type train_count: Int
    Return:
    -------
    :return: training_data, testing_data, training_labels, test_labels
    :rtype:  numpy.matrix, numpy.matrix, nd-array, nd-array
    """
    try:
        training_data = []
        testing_data = []
        training_label = []
        testing_label = []
        training_shape = []
        testing_shape = []
        for folder in sorted(os.listdir(directory), key=numerical_sort):
            path = directory + '/' + folder
            if os.path.isdir(path):
                files = os.listdir(path)
                if len(files) < train_count:
                    raise Exception("Number of required train samples is larger than the available samples.")
                counter = 0
                for i in range(1, len(files), 2):
                    image = imread(path + '/' + files[i])
                    if counter < train_count:
                        training_data.append(image.flatten())
                        training_label.append(folder)
                        training_shape.append(image.shape)
                        counter += 1
                    else:
                        testing_data.append(image.flatten())
                        testing_label.append(folder)
                        testing_shape.append(image.shape)
                for i in range(0, len(files), 2):
                    image = imread(path + '/' + files[i])
                    if counter < train_count:
                        training_data.append(image.flatten())
                        training_label.append(folder)
                        training_shape.append(image.shape)
                        counter += 1
                    else:
                        testing_data.append(image.flatten())
                        testing_label.append(folder)
                        testing_shape.append(image.shape)
    except Exception as e:
        print(e)
    else:
        return numpy.asmatrix(training_data), numpy.asarray(training_label), numpy.asarray(training_shape),\
               numpy.asmatrix(testing_data), numpy.asarray(testing_label), numpy.asarray(testing_shape)


if __name__ == '__main__':
    train_data, train_labels, train_shape, test_data, test_labels, test_shape = load(train_count=7)
