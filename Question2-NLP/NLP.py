import numpy as np


def gaussianInitialization(dimension):
    np.random.seed(50)
    p = np.random.normal(loc=0, scale=0.1, size=dimension)
    return p


def initializeWeights(layerSizes):
    WE = gaussianInitialization((layerSizes['inputSize'], layerSizes['embedSize']))
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

# TODO write an activation function

def sigmoid(z, condition):
    sig = 1 / (1 + np.exp(-z))
    if condition == 'forward':
        return sig
    if condition == 'gradient':
        return sig * (1 - sig)

# TODO look for softmax function and its derivative

def softmax(z, condition):
    if condition == 'normal':
        exp = np.exp(z)
        sm = exp / np.sum(exp)
        return sm
    if condition == 'stable':
        z = z - np.max(z, axis=-1, keepdims=True)
        exp = np.exp(z)
        sm = exp / np.sum(exp, axis=-1, keepdims=True)
        return sm
    if condition == 'log':
        z = z - np.max(z)
        l = np.logaddexp.reduce(z)
        sm = np.exp(z - l)
        return sm
    if condition == 'log stable':
        z = z - np.max(z)
        exp = np.exp(z)
        sm = np.log(exp / np.sum(z, axis=0))
        return sm


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
