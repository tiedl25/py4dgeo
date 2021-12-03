from py4dgeo.epoch import *

import numpy as np
import os
import pickle
import tempfile

from . import epoch1, find_data_file


def test_epoch_pickle(epoch1):

    # Operate in a temporary directory
    with tempfile.TemporaryDirectory() as dir:
        # Pickle the given KDTree
        fn = os.path.join(dir, "kdtree.pickle")
        with open(fn, "wb") as f:
            pickle.dump(epoch1, f)

        # Unpickle it
        with open(fn, "rb") as f:
            unpickled = pickle.load(f)

        # Try a radius search
        # result = unpickled.kdtree.radius_search(np.array([0, 0, 0]), 100)
        assert unpickled.cloud.shape[0] == epoch1.cloud.shape[0]


def test_as_epoch(epoch1):
    assert epoch1 is as_epoch(epoch1)
    assert np.allclose(epoch1.cloud, as_epoch(epoch1.cloud).cloud)


def test_read_from_xyz():
    epoch = read_from_xyz(find_data_file("ahk_2017_small.xyz"))
    assert np.isclose(np.max(epoch.cloud), 60.60600000002887)


def test_read_from_las():
    epoch = read_from_las(find_data_file("ahk_2017_small.las"))
    assert np.isclose(np.max(epoch.cloud), 146.24525)
