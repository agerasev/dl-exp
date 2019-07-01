import os
import urllib
import gzip
import struct

import numpy as np


dst = "data/"

def download():
    base = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    os.makedirs(dst, exist_ok=True)

    for file in files:
        if os.path.isfile(dst + file):
            print("file '%s' already exists" % file)
        else:
            print("downloading '%s' ..." % (base + file))
            urllib.request.urlretrieve(base + file, dst + file)

    print("done!")

def _read_labels(path):
    with gzip.open(path, "rb") as f:
        _magic, n = struct.unpack(">ll", f.read(2*4))
        assert(_magic == 2049)
        return np.frombuffer(f.read(n), dtype=np.dtype("B"))

def _read_images(path):
    with gzip.open(path, "rb") as f:
        _magic, n, sx, sy = struct.unpack(">llll", f.read(4*4))
        assert(_magic == 2051)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 1, sx, sy).astype(np.float64)/255.0
        return ((sx, sy), images)

def _read(prefix):
    labels = _read_labels(dst + prefix + "-labels-idx1-ubyte.gz")
    size, images = _read_images(dst + prefix + "-images-idx3-ubyte.gz")
    assert(labels.shape[0] == images.shape[0])
    return (size, labels, images)

def read_train():
    return _read("train")

def read_test():
    return _read("t10k")
