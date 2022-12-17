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

file = read_data(filePath='../Datasets/data3.h5')
keys = list(file.keys())

X_train = file[keys[0]][()]
y_train = file[keys[1]][()]
X_test = file[keys[2]][()]
y_test = file[keys[3]][()]

randomIndices = np.random.permutation(len(X_train))
X_train = X_train[randomIndices]
y_train = y_train[randomIndices]

print('\nThe shape of train input is:', X_train.shape)
print('The shape of test input is:', X_test.shape)
print('The shape of train output is:', y_train.shape)
print('The shape of test output is:', y_test.shape)

parameters = {'batchSize': 32,
              'learningRate': 1e-4,
              'momentumRate': 0.0,
              'epochNo': 50,
              'threshold': 15}

layerSizes = {'sampleSize': X_train.shape[0],
              'timeStepSize': X_train.shape[1],
              'featureSize': X_train.shape[2],
              'hiddenSize': 128,
              'outputSize': y_train.shape[1]}

print(layerSizes)
model = rnn.RNN(parameters, layerSizes)
train_acc, val_acc = model.fit(X_train, y_train, X_test, y_test)

plt.figure()
plt.title('Train accuracy')
plt.plot(train_acc)
plt.figure()
plt.title('Validation accuracy')
plt.plot(val_acc)
plt.show()

