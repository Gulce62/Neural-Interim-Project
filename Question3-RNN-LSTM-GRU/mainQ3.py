import h5py
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import RNN as rnn


def read_data(filePath):
    # Read the data with its path location
    try:
        file = h5py.File(filePath, 'r')
        print('File path is correct!!!')
        return file
    except Exception:
        print('Invalid file path!!!')
        sys.exit(1)

def trainValSplit(X_data, y_data): # TODO do it for every epoch
    np.random.seed(0)
    indicesNo = X_data.shape[0]
    randomIndicesVal = random.sample(range(indicesNo), int(indicesNo/10))
    randomIndicesTrain = list(set(range(indicesNo)) - set(randomIndicesVal))
    X_train = X_data[randomIndicesTrain, :, :]
    y_train = y_data[randomIndicesTrain, :]
    X_val = X_data[randomIndicesVal]
    y_val = y_data[randomIndicesVal]
    return X_train, y_train, X_val, y_val

file = read_data(filePath='../Datasets/data3.h5')
keys = list(file.keys())

X_trainAll = file[keys[0]][()]
y_trainAll = file[keys[1]][()]
X_test = file[keys[2]][()]
y_test = file[keys[3]][()]
X_train, y_train, X_val, y_val = trainValSplit(X_trainAll, y_trainAll)

print('\nThe shape of train input is:', X_train.shape)
print('The shape of validation input is:', X_val.shape)
print('The shape of test input is:', X_test.shape)
print('The shape of train output is:', y_train.shape)
print('The shape of validation output is:', y_val.shape)
print('The shape of test output is:', y_test.shape)

parameters = {'batchSize': 32,
              'learningRate': 0.1,
              'momentumRate': 0.85,
              'epochNo': 50,
              'threshold': 15}

layerSizes = {'sampleSize': X_train.shape[0],
              'timeStepSize': X_train.shape[1],
              'featureSize': X_train.shape[2],
              'hiddenSize': 128,
              'outputSize': y_train.shape[1]}

print(layerSizes)
rnnModel = rnn.RNN(parameters, layerSizes)
rnnModel.fit(X_train, y_train, X_val, y_val, layerSizes)



