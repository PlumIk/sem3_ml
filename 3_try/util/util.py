import random

import numpy as np

CLASS_NUMBER: int = 10
IMAGE_CHANNELS_COUNT: int = 1
IMAGE_SIZE: int = 28

Params = list[np.ndarray]

Outputs = list[np.ndarray]

Gradients = list[np.ndarray]

Moments = list[np.ndarray]


def create_filters(size, scale=1.0):
    stddev: float = scale / np.sqrt(np.array(size).prod())
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weights(size):
    return np.random.standard_normal(size=size) * 0.01


def shuffle(data, target):
    indices: list[int] = list(range(len(data)))
    random.shuffle(indices)
    return data[indices], target[indices]


def argmax2d(arr):
    idx: int = arr.argmax()
    return idx // arr.shape[1], idx % arr.shape[1]


def one_hot_coding(cls):
    code: np.ndarray = np.zeros(shape=(CLASS_NUMBER, 1))
    code[cls][0] = 1.0
    return code


def cross_entropy(p, q):
    return -(q * np.log(p)).sum()


def load_mnist():
    import os
    from keras.datasets import mnist
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    (train_data, train_target), (test_data, test_target) = mnist.load_data()

    def prepare(array: np.ndarray) -> np.ndarray:
        return array.astype(np.float32).reshape((len(array), IMAGE_CHANNELS_COUNT, IMAGE_SIZE, IMAGE_SIZE)) / 255

    return (prepare(train_data), train_target), (prepare(test_data), test_target)
