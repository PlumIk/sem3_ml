import gzip
import os
from urllib.request import urlretrieve

import numpy as np

URL = 'http://yann.lecun.com/exdb/mnist/'
FILES = ['train-images-idx3-ubyte.gz',
         'train-labels-idx1-ubyte.gz',
         't10k-images-idx3-ubyte.gz',
         't10k-labels-idx1-ubyte.gz']

def mnist():
    path = os.path.join(os.path.expanduser('~'), 'mnist')
    os.makedirs(path, exist_ok=True)
    for file in FILES:
        if file not in os.listdir(path):
            urlretrieve(URL + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def getImg(path):
        with gzip.open(path) as f:
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def getLables(path):
        with gzip.open(path) as f:
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)
        return integer_labels

    train_images = getImg(os.path.join(path, FILES[0]))
    train_labels = getLables(os.path.join(path, FILES[1]))
    test_images = getImg(os.path.join(path, FILES[2]))
    test_labels = getLables(os.path.join(path, FILES[3]))

    return train_images, train_labels, test_images, test_labels