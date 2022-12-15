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
    if condition == 'forward':
        tanh = np.tanh(z)
        return tanh
    if condition == 'gradient':
        return 1 - (z ** 2)


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
    timeStepSize = layerSizes['timeStepSize']
    caches = []
    A1_next = A1_0
    for timeStep in range(timeStepSize):
        currentTimeSample = X[:, timeStep, :]
        A1_next, A2, cache = forwardPassCell(currentTimeSample, A1_next, weights)
        caches.append(cache)
    return A2, caches


def crossEntropy(A2, y, condition):
    m = y.shape[0]
    if condition == 'loss':
        lossCE = -np.sum(y * np.log(A2)) / m
        return lossCE
    if condition == 'gradient':
        gradCE = A2 - y
        return gradCE


def backwardPassCell(dA1, weights, cache):
    X_timeStep = cache['X_timeStep']
    A1_next = cache['A1_next']
    A1_prev = cache['A1_prev']

    dZ1 = tanh(A1_next, 'gradient') * dA1
    dX_timeStep = np.dot(dZ1, weights['W1'].T)
    dW1 = np.dot(X_timeStep.T, dZ1)
    dWR = np.dot(A1_prev.T, dZ1)
    dA1_prev = np.dot(dZ1, weights['WR'].T)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    grad = {'dX_timeStep': dX_timeStep,
            'dW1': dW1,
            'dWR': dWR,
            'dA1_prev': dA1_prev,
            'db1': db1}
    return grad


def backwardPass(X, y, A2, weights, caches, layerSizes):
    sampleSize = X.shape[0]
    timeStepSize = layerSizes['timeStepSize']
    featureSize = layerSizes['featureSize']
    hiddenSize = layerSizes['hiddenSize']

    cache = caches[0]['A1_next']
    dZ2 = crossEntropy(A2, y, 'gradient')
    dW2 = (1 / sampleSize) * np.dot(cache.T, dZ2)
    db2 = (1 / sampleSize) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = (1 / sampleSize) * np.dot(dZ2, weights['W2'].T)

    dX = np.zeros((sampleSize, timeStepSize, featureSize))
    dW1 = np.zeros((featureSize, hiddenSize))
    dWR = np.zeros((hiddenSize, hiddenSize))
    dA1_prev = np.zeros((sampleSize, hiddenSize))
    db1 = np.zeros((1, hiddenSize))

    for timeStep in reversed(range(timeStepSize)):
        grads = backwardPassCell(dA1, weights, caches[timeStep])
        dX[:, timeStep, :] = grads["dX_timeStep"]
        dW1 += grads["dW1"]
        dWR += grads["dWR"]
        db1 += grads["db1"]
        dA1_prev = grads["dA1_prev"]
    dA1_0 = dA1_prev

    J_grad = {'dW2': dW2,
              'db2': db2,
              'dX': dX,
              'dW1': dW1,
              'dWR': dWR,
              'db1': db1,
              'dA1_0': dA1_0}
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

    return weights, prevWeights


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

    def fit(self, X_train, y_train, X_val, y_val, layerSizes):
        self.weights = initializeWeights(layerSizes)

        m = layerSizes['sampleSize']
        if m % self.batchSize == 0:
            iterationNo = m // self.batchSize
        else:
            iterationNo = m // self.batchSize + 1

        for epoch in range(self.epochNo):
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
                A1_0 = np.zeros((X_batch.shape[0], layerSizes['hiddenSize']))
                A2, caches = forwardPass(X_batch, A1_0, self.weights, layerSizes)
                J_grad = backwardPass(X_batch, y_batch, A2, self.weights, caches, layerSizes)
                self.weights, self.prevWeights = updateParameters(self.weights, self.prevWeights, J_grad,
                                                                  self.learningRate, self.momentumRate)
                J_train = crossEntropy(A2, y_batch, 'loss')
                print(J_train)



    def predict(self, X):
        A1_0 = np.zeros((X.shape[0], self.layerSizes['hiddenSize']))
        A2, caches = forwardPass(X, A1_0, self.weights, self.layerSizes)
        y_pred = np.argmax(A2)
        return y_pred

