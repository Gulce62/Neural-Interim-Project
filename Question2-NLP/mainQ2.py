import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import NLP as nlp
import OneHotEncoder as ohe
import time

t1 = time.time()
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

print('\nThe shape of train input is:', X_train.shape)
print('The shape of validation input is:', X_val.shape)
print('The shape of test input is:', X_test.shape)
print('The shape of train output is:', y_train.shape)
print('The shape of validation output is:', y_val.shape)
print('The shape of test output is:', y_test.shape)
print('The shape of the vocabulary is:', words.shape)

encode = ohe.OneHotEncoder(words.shape[0])

X_trainOHE = encode.vectorize(X_train)
X_trainOHE = encode.feedInput(X_trainOHE, 'concatenate')
X_valOHE = encode.vectorize(X_val)
X_valOHE = encode.feedInput(X_valOHE, 'concatenate')
X_testOHE = encode.vectorize(X_test)
X_testOHE = encode.feedInput(X_testOHE, 'concatenate')
y_trainOHE = encode.vectorize(y_train.reshape(y_train.shape[0], 1)).squeeze()
y_valOHE = encode.vectorize(y_val.reshape(y_val.shape[0], 1)).squeeze()
y_testOHE = encode.vectorize(y_test.reshape(y_test.shape[0], 1)).squeeze()

print('\nThe shape of one-hot encoded train input is:', X_trainOHE.shape)
print('The shape of one-hot encoded validation input is:', X_valOHE.shape)
print('The shape of one-hot encoded test input is:', X_testOHE.shape)
print('The shape of one-hot encoded train output is:', y_trainOHE.shape)
print('The shape of one-hot encoded validation output is:', y_valOHE.shape)
print('The shape of one-hot encoded test output is:', y_testOHE.shape, '\n')

D = [32, 16, 8]
P = [256, 128, 64]
#D = [32]
#P = [256]

parameters = {'batchSize': 200,
              'learningRate': 0.15,
              'momentumRate': 0.85,
              'epochNo': 50,
              'threshold': 0.1}

for d in D:
    for p in P:
        layerSizes = {'featureSize': X_train.shape[1] * words.shape[0],
                      'sampleSize': X_train.shape[0],
                      'embedSize': d,
                      'hiddenSize': p,
                      'outputSize': words.shape[0]}

        nlpModel = nlp.NLP(parameters, layerSizes)
        train_acc, val_acc = nlpModel.fit(X_trainOHE, y_trainOHE, X_valOHE, y_valOHE, layerSizes)
        plt.figure()
        plt.title('Train accuracy for d = '+str(d)+' p = '+str(p))
        plt.plot(train_acc)
        plt.figure()
        plt.title('Validation accuracy for d = ' + str(d) + ' p = ' + str(p))
        plt.plot(val_acc)
        plt.show()

        t2 = time.time()
        print(t2-t1)

t3 = time.time()
print(t3-t1)
