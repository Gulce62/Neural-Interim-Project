import h5py
import sys
import numpy as np
import pandas as pd
import seaborn as sn
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


def getConfusionMatrix(y_true, y_pred):
    plt.figure()
    # Get the confusion matrix
    y_true = pd.Categorical(y_true.ravel())
    y_pred = pd.Categorical(y_pred.ravel())
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    sn.heatmap(confusion_matrix, cmap="Blues", annot=True)
    return confusion_matrix


def accuracyScore(y_true, y_pred):
    accuracyBool = (y_true.ravel() == y_pred.ravel())
    accuracy = np.count_nonzero(accuracyBool) / accuracyBool.shape[0]
    return accuracy


file = read_data(filePath='../Datasets/data3.h5')
keys = list(file.keys())

X_train = file[keys[0]][()]
y_train = file[keys[1]][()]
X_test = file[keys[2]][()]
y_test = file[keys[3]][()]

randomIndices = np.random.permutation(len(X_train))
X_train = X_train[randomIndices]
y_train = y_train[randomIndices]

slice = int(X_train.shape[0] / 10)
X_val = X_train[0:slice, :, :]
y_val = y_train[0:slice, :]

print('\nThe shape of train input is:', X_train.shape)
print('The shape of test input is:', X_test.shape)
print('The shape of train output is:', y_train.shape)
print('The shape of test output is:', y_test.shape)

parameters = {'batchSize': 16,
              'learningRate': 1e-4,
              'momentumRate': 0.0,
              'epochNo': 50,
              'threshold': 0.8}

layerSizes = {'sampleSize': X_train.shape[0],
              'timeStepSize': X_train.shape[1],
              'featureSize': X_train.shape[2],
              'hiddenSize': 128,
              'outputSize': y_train.shape[1]}

print(layerSizes)
model = rnn.RNN(parameters, layerSizes)
train_acc, val_acc = model.fit(X_train, y_train, X_val, y_val)

plt.figure()
plt.title('Train accuracy vs. epoch')
plt.plot(train_acc)
plt.figure()
plt.title('Validation accuracy vs. epoch')
plt.plot(val_acc)
plt.show()

pred, pred_index = model.predict(X_train)
true_index = np.argmax(y_test, axis=1)
train_accuracy = accuracyScore(true_index, pred_index)
print('Train accuracy is:', train_accuracy)
confusion_matrix_train = getConfusionMatrix(true_index, pred_index)

pred, pred_index = model.predict(X_test)
true_index = np.argmax(y_test, axis=1)
test_accuracy = accuracyScore(true_index, pred_index)
print('Test accuracy is:', test_accuracy)
confusion_matrix_test = getConfusionMatrix(true_index, pred_index)
