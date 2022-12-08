import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import NLP as nlp


def read_data(filePath):
    # Read the data with its path location
    try:
        file = h5py.File(filePath, 'r')
        print('File path is correct!!!')
        return file
    except Exception:
        print('Invalid file path!!!')
        sys.exit(1)


file = read_data(filePath='../Datasets/data2.h5')
keys = list(file.keys())
X_train = file[keys[3]][()]
y_train = file[keys[2]][()]
X_val = file[keys[5]][()]
y_val = file[keys[4]][()]
X_test = file[keys[1]][()]
y_test = file[keys[0]][()]
words = file[keys[6]][()]

print('The shape of train input is:', X_train.shape)
print('The shape of validation input is: ', X_val.shape)
print('The shape of test input is:', X_test.shape)
print('The shape of train output is:', y_train.shape)
print('The shape of validation output is:', y_val.shape)
print('The shape of test output is:', y_test.shape)

print('The shape of the vocabulary is:', words.shape)

batchSize = 200
learningRate = 0.15
momentumRate = 0.85
epochNo = 50
D = 32
P = 256
# D = [32, 16, 8]
# P = [256, 128, 64]

parameters = {'batchSize': batchSize,
              'learningRate': learningRate,
              'momentumRate': momentumRate,
              'epochNo': epochNo}

layerSizes = {'featureSize': X_train.shape[1],
              'inputSize': X_train.shape[0],
              'embedSize': D,
              'hiddenSize': P,
              'outputSize': words.shape[0]}

weights = nlp.initializeWeights(layerSizes)


print(X_train[0])










