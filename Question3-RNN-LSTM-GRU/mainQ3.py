import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt


def read_data(filePath):
    # Read the data with its path location
    try:
        file = h5py.File(filePath, 'r')
        print('File path is correct!!!')
        return file
    except Exception:
        print('Invalid file path!!!')
        sys.exit(1)

file = h5py.File('data3.h5', 'r')
keys = list(file.keys())
print(keys)