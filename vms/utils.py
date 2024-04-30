import msl 
import h5py
import numpy as np
from os.path import join

def get_dataset_location():

    dataset_location = msl.__file__
    dataset_location = dataset_location.replace('__init__.py', '')
    dataset_location = join(dataset_location[:-4], "dataset")

    return dataset_location

def load_mat(mat_filename, key):
    with h5py.File(mat_filename, 'r') as f:
        return np.array(f[key])

def is_symmetric(A, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)