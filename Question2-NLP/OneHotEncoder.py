import numpy as np


class OneHotEncoder:
    def __init__(self, wordSize):
        self.wordSize = wordSize

    def vectorize(self, data):
        OHEVector = np.zeros((data.shape[0], data.shape[1], self.wordSize))
        for feature in range(data.shape[1]):
            indices = (data[..., feature] - 1)
            for sample in range(indices.shape[0]):
                OHEVector[sample, feature, indices[sample]] = 1
        return OHEVector

    def feedInput(self, data, condition):
        if condition == 'sum':
            inputData = np.sum(data, axis=1)
            return inputData
        if condition == 'concatenate':
            inputData = np.concatenate((data[:, 0, :], data[:, 1, :], data[:, 2, :]), axis=1)
            return inputData
