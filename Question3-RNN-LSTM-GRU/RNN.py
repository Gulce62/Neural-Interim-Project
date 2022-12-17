import numpy as np
import matplotlib.pyplot as plt


def initialization(Lpre, Lpost, size):
    wo = np.sqrt(6 / (Lpre + Lpost))
    parameter = np.random.uniform(-wo, wo, size=size)
    return parameter


def initializeWeights(layerSizes):
    featureSize = layerSizes['featureSize']
    hiddenSize = layerSizes['hiddenSize']
    outputSize = layerSizes['outputSize']
    W1 = initialization(featureSize, hiddenSize, (featureSize, hiddenSize))
    WR = initialization(hiddenSize, hiddenSize, (hiddenSize, hiddenSize))
    W2 = initialization(hiddenSize, outputSize, (hiddenSize, outputSize))
    b1 = initialization(featureSize, hiddenSize, (1, hiddenSize))
    b2 = initialization(hiddenSize, outputSize, (1, outputSize))
    We = {'W1': W1,
          'W2': W2,
          'WR': WR,
          'b1': b1,
          'b2': b2}
    return We


def tanh(z, condition):
    if condition == 'forward':
        tanh = np.tanh(z)
        return tanh
    if condition == 'gradient':
        tanhGrad = 1 - (z ** 2)
        return tanhGrad


def softmax(z):
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    sm = exp / np.sum(exp, axis=-1, keepdims=True)
    return sm


def forwardPassCell(currentTimeStep, A1_prev, weights):
    W1 = weights['W1']
    WR = weights['WR']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']

    current_Z1 = np.dot(currentTimeStep, W1) + np.dot(A1_prev, WR) + b1
    current_A1 = tanh(current_Z1, 'forward')
    current_Z2 = np.dot(current_A1, W2) + b2
    current_A2 = softmax(current_Z2)

    cache = {'currentTimeStep': currentTimeStep,
             'A1_next': current_A1,
             'A2': current_A2}
    return cache


def forwardPass(X, weights, layerSizes):
    input_timeStep = {}
    hidden_timeStep = {}
    output_timeStep = {}

    A1_prev = np.zeros((1, layerSizes['hiddenSize']))
    for timeStep in range(layerSizes['timeStepSize']):
        cache = forwardPassCell(X[:, timeStep, :], A1_prev, weights)

        input_timeStep[timeStep] = cache['currentTimeStep']
        hidden_timeStep[timeStep] = cache['A1_next']
        output_timeStep[timeStep] = cache['A2']

        A1_prev = cache['A1_next']

    return input_timeStep, hidden_timeStep, output_timeStep


def crossEntropyLoss(y_true, y_pred):
    m = y_pred.shape[0]
    lossCE = -np.sum(y_true * np.log(y_pred)) / m
    return lossCE


def backwardPassCell(dA1, currentTimeStep, A1_next, A1_prev, weights):
    dZ1 = tanh(A1_next, 'gradient') * dA1
    dW1 = np.dot(currentTimeStep.T, dZ1)
    dWR = np.dot(A1_prev.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    dA1_prev = np.dot(dZ1, weights['WR'].T)

    grad = {'dW1': dW1,
            'dWR': dWR,
            'db1': db1,
            'dA1_prev': dA1_prev}
    return grad


def backwardPass(y, cache, weights, layerSizes):
    input_timeStep, hidden_timeStep, output_timeStep = cache

    sampleSize = y.shape[0]
    timeStepSize = layerSizes['timeStepSize']
    featureSize = layerSizes['featureSize']
    hiddenSize = layerSizes['hiddenSize']
    outputSize = layerSizes['outputSize']

    dW1 = np.zeros((featureSize, hiddenSize))
    dWR = np.zeros((hiddenSize, hiddenSize))
    dW2 = np.zeros((hiddenSize, outputSize))
    db1 = np.zeros((1, hiddenSize))
    db2 = np.zeros((1, outputSize))
    dA1_prev = np.zeros((sampleSize, hiddenSize))

    dA2 = np.copy(output_timeStep[149])
    length = len(y)
    index = np.argmax(y, axis=1)
    dA2[np.arange(length), index] -= 1

    dW2 = np.dot(hidden_timeStep[149].T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)

    for timeStep in reversed(range(1, timeStepSize)):
        dA1 = np.dot(dA2, weights['W2'].T) + dA1_prev
        grads = backwardPassCell(dA1, input_timeStep[timeStep], hidden_timeStep[timeStep],
                                 hidden_timeStep[timeStep-1], weights)
        dW1 += grads['dW1']
        dWR += grads['dWR']
        db1 += grads['db1']
        dA1_prev = grads['dA1_prev']

    weightList = [dW1, db1, dWR, dW2, db2]
    for weightDerivatives in weightList:
        np.clip(weightDerivatives, -10, 10, out=weightDerivatives)

    J_grad = {'dW2': dW2,
              'db2': db2,
              'dW1': dW1,
              'dWR': dWR,
              'db1': db1}
    return J_grad


def updateParameters(weights, prevWeights, J_grad, learningRate, momentumRate):
    prevWeights['dWR_prev'] = learningRate * J_grad['dWR'] + momentumRate * prevWeights['dWR_prev']
    prevWeights['dW1_prev'] = learningRate * J_grad['dW1'] + momentumRate * prevWeights['dW1_prev']
    prevWeights['dW2_prev'] = learningRate * J_grad['dW2'] + momentumRate * prevWeights['dW2_prev']
    prevWeights['db1_prev'] = learningRate * J_grad['db1'] + momentumRate * prevWeights['db1_prev']
    prevWeights['db2_prev'] = learningRate * J_grad['db2'] + momentumRate * prevWeights['db2_prev']

    weights['WR'] -= prevWeights['dWR_prev']
    weights['W1'] -= prevWeights['dW1_prev']
    weights['W2'] -= prevWeights['dW2_prev']
    weights['b1'] -= prevWeights['db1_prev']
    weights['b2'] -= prevWeights['db2_prev']

    learningRate *= 0.9999
    return weights, prevWeights, learningRate


def getAccuracy(y_true, y_pred):
    accuracyBool = (y_true.ravel() == y_pred.ravel())
    accuracy = np.count_nonzero(accuracyBool) / accuracyBool.shape[0]
    return accuracy


class RNN:
    def __init__(self, parameters, layerSizes):
        self.batchSize = parameters['batchSize']
        self.learningRate = parameters['learningRate']
        self.momentumRate = parameters['momentumRate']
        self.epochNo = parameters['epochNo']
        self.threshold = parameters['threshold']

        self.layerSizes = layerSizes
        self.weights = None
        self.prevWeights = {'dWR_prev': 0,
                            'dW1_prev': 0,
                            'dW2_prev': 0,
                            'db1_prev': 0,
                            'db2_prev': 0}

    def fit(self, X_train, y_train, X_val, y_val):
        np.random.seed(150)
        self.weights = initializeWeights(self.layerSizes)

        m = self.layerSizes['sampleSize']
        if m % self.batchSize == 0:
            iterationNo = m // self.batchSize
        else:
            iterationNo = m // self.batchSize + 1

        train_acc = []
        val_acc = []
        for epoch in range(self.epochNo):
            for batch in range(iterationNo):
                startIdx = batch * self.batchSize
                endIdx = startIdx + self.batchSize
                if batch == iterationNo - 1:
                    X_batch = X_train[startIdx:]
                    y_batch = y_train[startIdx:]
                else:
                    X_batch = X_train[startIdx:endIdx]
                    y_batch = y_train[startIdx:endIdx]
                cache = forwardPass(X_batch, self.weights, self.layerSizes)
                J_grad = backwardPass(y_batch, cache, self.weights, self.layerSizes)
                self.weights, self.prevWeights, self.learningRate = updateParameters(self.weights, self.prevWeights,
                                                                                     J_grad, self.learningRate, self.momentumRate)
            print('\n=========================== Epoch', epoch, 'finished ============================')

            y_pred, pred_index = self.predict(X_train)
            J_train = crossEntropyLoss(y_train, y_pred)
            true_index = np.argmax(y_train, axis=1)
            train_accuracy = getAccuracy(true_index, pred_index)
            train_acc.append(train_accuracy)

            y_pred, pred_index = self.predict(X_val)
            J_val = crossEntropyLoss(y_val, y_pred)
            true_index = np.argmax(y_val, axis=1)
            val_accuracy = getAccuracy(true_index, pred_index)
            val_acc.append(val_accuracy)

            print('Training loss for epoch', epoch, 'is:', J_train)
            print('Validation loss for epoch', epoch, 'is:', J_val)
            print('Train accuracy is', train_accuracy)
            print('Validation accuracy is', val_accuracy)
            print('Difference between training and validation loss:', np.abs(J_train - J_val))
            print('=========================================================================')
            if np.abs(J_train - J_val) > self.threshold:
                print('Finish training!!!')
                break
        return train_acc, val_acc

    def predict(self, X):
        cache = forwardPass(X, self.weights, self.layerSizes)
        y_pred = cache[2][149]
        y_predIndex = np.argmax(y_pred, axis=1)
        return y_pred, y_predIndex
