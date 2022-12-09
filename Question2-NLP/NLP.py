import numpy as np


def gaussianInitialization(dimension):
    np.random.seed(8)
    p = np.random.normal(loc=0, scale=0.1, size=dimension)
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


# TODO look for softmax function and its derivative

def softmax(z, condition):
    exp = np.exp(z - np.max(z))
    sm = exp / np.sum(exp)
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
    Z1 = np.dot(emb, W1) + b1
    A1 = sigmoid(Z1, 'forward')
    Z2 = np.dot(A1, W2) + b2
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
        lossCE = np.sum(-y * np.log(A2)) / m
        return lossCE
    if condition == 'gradient':
        gradCE = -y / A2
        return gradCE


def backwardPass(X, y, weights, cache):
    m = X.shape[0]

    emb = cache['emb']
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']

    dZ2 = crossEntropy(A2, y, 'gradient') * softmax(Z2, 'gradient')
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


def nlpCost(X, weights, y):
    cache = forwardPass(X, weights)
    J = crossEntropy(cache['A2'], y, 'loss')
    J_grad = backwardPass(X, y, weights, cache)
    return J, J_grad

def predict(X, y, weights):
    cache = forwardPass(X, weights)
    y_pred = cache['A2']
    print(y_pred[0])
    print(y[0])
    print(np.where(y[2]==1))
    print(y_pred[2].argsort()[-50:][::-1])
    print(-y_pred[2].argsort()[:50])

class NLP:
    def __init__(self, weights, parameters, layerSizes):
        self.WE = weights['WE']
        self.W1 = weights['W1']
        self.W2 = weights['W2']
        self.b1 = weights['b1']
        self.b2 = weights['b2']

        self.batchSize = parameters['batchSize']
        self.learningRate = parameters['learningRate']
        self.momentumRate = parameters['momentumRate']
        self.epochNo = parameters['epochNo']

        self.featureSize = layerSizes['featureSize']
        self.inputSize = layerSizes['inputSize']
        self.embedSize = layerSizes['embedSize']
        self.hiddenSize = layerSizes['hiddenSize']
        self.outputSize = layerSizes['outputSize']
