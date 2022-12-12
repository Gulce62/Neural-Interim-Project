import numpy as np


def gaussianInitialization(dimension):
    np.random.seed(0)
    p = np.random.normal(loc=0, scale=0.01, size=dimension)
    return p


def initializeWeights(layerSizes):
    WE = gaussianInitialization((layerSizes['featureSize'], layerSizes['embedSize']))
    W1 = gaussianInitialization((layerSizes['embedSize'], layerSizes['hiddenSize']))
    W2 = gaussianInitialization((layerSizes['hiddenSize'], layerSizes['outputSize']))
    b1 = gaussianInitialization((1, layerSizes['hiddenSize']))
    b2 = gaussianInitialization((1, layerSizes['outputSize']))

    weights = {'WE': WE,
               'W1': W1,
               'W2': W2,
               'b1': b1,
               'b2': b2}
    return weights


def sigmoid(z, condition):
    sig = 1 / (1 + np.exp(-z))
    if condition == 'forward':
        return sig
    if condition == 'gradient':
        return sig * (1 - sig)


def softmax(z, condition):
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    sm = exp / np.sum(exp, axis=1, keepdims=True)
    if condition == 'forward':
        return sm
    if condition == 'gradient':
        return sm * (1 - sm)


def forwardPass(X, weights):
    WE = weights['WE']
    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']

    emb = np.dot(X, WE)
    Z1 = np.dot(emb, W1) - b1
    A1 = sigmoid(Z1, 'forward')
    Z2 = np.dot(A1, W2) - b2
    A2 = softmax(Z2, 'forward')

    cache = {'emb': emb,
             'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}
    return cache


def crossEntropy(A2, y, condition):
    m = y.shape[0]
    if condition == 'loss':
        lossCE = -np.sum(y * np.log(A2)) / m
        return lossCE
    if condition == 'gradient':
        gradCE = A2 - y  # TODO look for the gradient of cross entropy
        return gradCE


def backwardPass(X, y, weights, cache):
    m = X.shape[0]

    emb = cache['emb']
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']

    dZ2 = crossEntropy(A2, y, 'gradient')
    dW2 = (1 / m) * (np.dot(A1.T, dZ2))
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = (np.dot(dZ2, weights['W2'].T)) * sigmoid(Z1, 'gradient')
    dW1 = (1 / m) * np.dot(emb.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    demb = (np.dot(dZ1, weights['W1'].T))
    dWE = (1 / m) * np.dot(X.T, demb)

    grads = {'dWE': dWE,
             'dW1': dW1,
             'db1': db1,
             'dW2': dW2,
             'db2': db2}
    return grads


def updateParameters(weights, prevWeights, J_grad, learningRate, momentumRate):  # TODO add momentum
    dWE = learningRate * J_grad['dWE'] + momentumRate * prevWeights['dWE_prev']
    dW1 = learningRate * J_grad['dW1'] + momentumRate * prevWeights['dW1_prev']
    dW2 = learningRate * J_grad['dW2'] + momentumRate * prevWeights['dW2_prev']
    db1 = learningRate * J_grad['db1'] + momentumRate * prevWeights['db1_prev']
    db2 = learningRate * J_grad['db2'] + momentumRate * prevWeights['db2_prev']

    weights['WE'] -= dWE
    weights['W1'] -= dW1
    weights['W2'] -= dW2
    weights['b1'] -= db1
    weights['b2'] -= db2

    prevWeights['dWE_prev'] = dWE
    prevWeights['dW1_prev'] = dW1
    prevWeights['dW2_prev'] = dW2
    prevWeights['db1_prev'] = db1
    prevWeights['db2_prev'] = db2

    return weights, prevWeights


class NLP:
    def __init__(self, parameters, layerSizes):
        self.batchSize = parameters['batchSize']
        self.learningRate = parameters['learningRate']
        self.momentumRate = parameters['momentumRate']
        self.epochNo = parameters['epochNo']

        self.layerSizes = layerSizes
        self.weights = None
        self.prevWeights = {'dWE_prev': 0,
                            'dW1_prev': 0,
                            'dW2_prev': 0,
                            'db1_prev': 0,
                            'db2_prev': 0}

    def fit(self, X_train, y_train, layerSizes):
        self.weights = initializeWeights(layerSizes)

        m = layerSizes['inputSize']
        if m % self.batchSize == 0:
            iterationNo = m // self.batchSize
        else:
            iterationNo = m // self.batchSize + 1

        for epoch in range(self.epochNo):
            J = 0
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
            print('Loss for epoch', epoch, 'is:', J / iterationNo)

    def predict(self, X):
        cache = forwardPass(X, self.weights)
        y_pred = cache['A2']
        pred_index = np.argmax(y_pred, axis=1)
        return pred_index
