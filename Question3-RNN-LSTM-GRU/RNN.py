import numpy as np
import matplotlib.pyplot as plt


def initialization(Lpre, Lpost):
    np.random.seed(0)  # TODO look for different seed values
    wo = np.sqrt(6 / (Lpre + Lpost))
    parameter = np.random.uniform(-wo, wo, size=(Lpre, Lpost))
    return parameter


def initializeWeights(layersizes):
    W1 = initialization(layersizes['featureSize'], layersizes['hiddenSize'])
    WR = initialization(layersizes['hiddenSize'], layersizes['hiddenSize'])
    W2 = initialization(layersizes['hiddenSize'], layersizes['outputSize'])
    b1 = initialization(1, layersizes['hiddenSize'])
    b2 = initialization(1, layersizes['outputSize'])
    We = {'W1': W1,
          'W2': W2,
          'WR': WR,
          'b1': b1,
          'b2': b2}
    return We


def tanh(z, condition):
    tanh = np.tanh(z)
    if condition == 'forward':
        return tanh
    if condition == 'gradient':
        return 1-(tanh**2)


def softmax(z, condition):
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    sm = exp / np.sum(exp, axis=1, keepdims=True)
    if condition == 'forward':
        return sm
    if condition == 'gradient':
        return sm * (1 - sm)


def forwardPassCell(X_timeStep, A1_prev, weights):
    W1 = weights['W1']
    WR = weights['WR']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']

    Z1_next = np.dot(X_timeStep, W1) + np.dot(A1_prev, WR) - b1
    A1_next = tanh(Z1_next, 'forward')
    Z2 = np.dot(A1_next, W2) - b2
    A2 = softmax(Z2, 'forward')

    cache = {'X_timeStep': X_timeStep,
             'A1_next': A1_next,
             'A1_prev': A1_prev}
    return A1_next, A2, cache


def forwardPass(X, A1_0, weights, layerSizes):
    sampleSize = layerSizes['sampleSize']
    timeStepSize = layerSizes['timeStepSize']
    hiddenSize = layerSizes['hiddenSize']
    outputSize = layerSizes['outputSize']

    cacheList = []
    hiddenUnits = np.zeros((sampleSize, timeStepSize, hiddenSize))
    outputUnits = np.zeros((sampleSize, timeStepSize, outputSize))

    A1_next = A1_0
    for timeStep in range(timeStepSize):
        currentTimeSample = X[:, timeStep, :]
        A1_next, A2, cache = forwardPassCell(currentTimeSample, A1_next, weights)
        hiddenUnits[:, timeStep, :] = A1_next
        outputUnits[:, timeStep, :] = A2
        cacheList.append(cache)

    caches = {'cacheList': cacheList,
              'X': X,
              'y': A2}
    return hiddenUnits, outputUnits, caches


def crossEntropy(A2, y, condition):
    m = y.shape[0]
    if condition == 'loss':
        lossCE = -np.sum(y * np.log(A2)) / m
        return lossCE
    if condition == 'gradient':
        gradCE = A2 - y
        return gradCE


def backwardPassCell():
    pass


def backwardPass():
    pass


class RNN:
    def __init__(self, parameters, layerSizes):
        self.batchSize = parameters['batchSize']
        self.learningRate = parameters['learningRate']
        self.momentumRate = parameters['momentumRate']
        self.epochNo = parameters['epochNo']
        self.threshold = parameters['threshold']

        self.layerSizes = layerSizes
        self.weights = None
        self.prevWeights = {'dWE_prev': 0,
                            'dW1_prev': 0,
                            'dW2_prev': 0,
                            'db1_prev': 0,
                            'db2_prev': 0}

    def fit(self, X_train, y_train, X_val, y_val, layerSizes):
        self.weights = initializeWeights(layerSizes)
        A1_0 = np.zeros((1, layerSizes['hiddenSize']))
        hiddenUnits, outputUnits, caches = forwardPass(X_train, A1_0, self.weights, layerSizes)
        J = crossEntropy(caches['y'], y_train, 'loss')
        print(J)

        """
        m = layerSizes['inputSize']
        if m % self.batchSize == 0:
            iterationNo = m // self.batchSize
        else:
            iterationNo = m // self.batchSize + 1

        for epoch in range(self.epochNo):
            J = 0
            train_acc = []
            val_acc = []
            for batch in range(iterationNo):
                startIdx = batch * self.batchSize
                endIdx = startIdx + self.batchSize
                if batch == iterationNo - 1:
                    X_batch = X_train[startIdx:]
                    y_batch = y_train[startIdx:]
                else:
                    X_batch = X_train[startIdx:endIdx]
                    y_batch = y_train[startIdx:endIdx]
                cache = forwardPass(X_batch, self.weights)
                J += crossEntropy(cache['A2'], y_batch, 'loss')
                J_grad = backwardPass(X_batch, y_batch, self.weights, cache)
                self.weights, self.prevWeights = updateParameters(self.weights, self.prevWeights, J_grad,
                                                                  self.learningRate, self.momentumRate)
            print('Epoch', epoch, 'finished ==================>')

            y_pred, pred_index = self.predict(X_train)
            J_train = crossEntropy(y_pred, y_train, 'loss')
            print('Training loss for epoch', epoch, 'is:', J_train)
            true_index = np.argmax(y_train, axis=1)
            train_accuracy = getAccuracy(true_index, pred_index)
            train_acc.append(train_accuracy)
            print('Train accuracy is', train_accuracy)

            y_pred, pred_index = self.predict(X_val)
            J_val = crossEntropy(y_pred, y_val, 'loss')
            print('Validation loss for epoch', epoch, 'is:', J_val)
            true_index = np.argmax(y_val, axis=1)
            val_accuracy = getAccuracy(true_index, pred_index)
            val_acc.append(val_accuracy)
            print('Validation accuracy is', val_accuracy)

            print('Difference between training and validation loss:', np.abs(J_train - J_val))
            if np.abs(J_train - J_val) > self.threshold:
                print('Finish training!!!')
                break
        return train_acc, val_acc"""

