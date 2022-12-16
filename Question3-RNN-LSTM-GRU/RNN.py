import numpy as np
import matplotlib.pyplot as plt


def initialization(Lpre, Lpost):
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
        tanhGrad = 1 - (z ** 2)
        return tanhGrad


def softmax(z):
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    sm = exp / np.sum(exp, axis=-1, keepdims=True)
    return sm



def forwardPassCell(X_timeStep, A1_prev, weights):
    W1 = weights['W1']
    WR = weights['WR']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']

    Z1_next = np.dot(X_timeStep, W1) + np.dot(A1_prev, WR) - b1
    A1_next = tanh(Z1_next, 'forward')
    Z2 = np.dot(A1_next, W2) - b2
    A2 = softmax(Z2)

    cache = {'X_timeStep': X_timeStep,
             'A1_next': A1_next,
             'A2': A2}
    return A1_next, Z2, cache


def forwardPass(X, weights, layerSizes):
    X_timeStep = dict()
    A1_next = dict()
    A2 = dict()
    A1_prev = np.zeros((X.shape[0], layerSizes['hiddenSize']))
    for timeStep in range(layerSizes['timeStepSize']):
        currentTimeSample = X[:, timeStep, :]
        A1_prev, Z2, cache = forwardPassCell(currentTimeSample, A1_prev, weights)
        X_timeStep[timeStep] = cache['X_timeStep']
        A1_next[timeStep] = cache['A1_next']
        A2[timeStep] = cache['A2']
    return X_timeStep, A1_next, A2


def crossEntropy(A2, y):
    m = A2.shape[0]
    lossCE = -np.sum(y * np.log(A2)) / m
    return lossCE


def backwardPassCell(dA1, weights, X_timeStep, A1_next, A1_prev):
    dZ1 = tanh(A1_next, 'gradient') * dA1
    dW1 = np.dot(X_timeStep.T, dZ1)
    dWR = np.dot(A1_prev.T, dZ1)
    dA1_prev = np.dot(dZ1, weights['WR'].T)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    grad = {'dW1': dW1,
            'dWR': dWR,
            'dA1_prev': dA1_prev,
            'db1': db1}
    return grad


def backwardPass(X, y, weights, caches, layerSizes):
    sampleSize = X.shape[0]
    timeStepSize = layerSizes['timeStepSize']
    featureSize = layerSizes['featureSize']
    hiddenSize = layerSizes['hiddenSize']
    outputSize = layerSizes['outputSize']

    dW2 = np.zeros((hiddenSize, outputSize))
    dWR = np.zeros((hiddenSize, hiddenSize))
    dW1 = np.zeros((featureSize, hiddenSize))
    db2 = np.zeros((1, outputSize))
    db1 = np.zeros((1, hiddenSize))
    dA1_prev = np.zeros((sampleSize, hiddenSize))

    X_timeStep, A1_next, A2 = caches

    dA2 = A2[149]
    dA2[np.arange(len(y)), np.argmax(y, 1)] -= 1

    dW2 += np.dot(A1_next[149].T, dA2)
    db2 += np.sum(dA2, axis=0, keepdims=True)

    for timeStep in reversed(range(1, timeStepSize)):
        dA1 = np.dot(dA2, weights['W2'].T) + dA1_prev
        grads = backwardPassCell(dA1, weights, X_timeStep[timeStep], A1_next[timeStep], A1_next[timeStep-1])
        dW1 += grads['dW1']
        dWR += grads['dWR']
        db1 += grads['db1']
        dA1_prev = grads['dA1_prev']

    for grad in [dW1, dWR, dW2, db1, db2]:
        np.clip(grad, -10, 10, out=grad)

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
        np.random.seed(150)
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
                caches = forwardPass(X_batch, self.weights, layerSizes)
                J_grad = backwardPass(X_batch, y_batch, self.weights, caches, layerSizes)
                self.weights, self.prevWeights = updateParameters(self.weights, self.prevWeights, J_grad,
                                                                  self.learningRate, self.momentumRate)

            cross_loss_train = crossEntropy(caches[2][149], y_batch)
            predictions_train = self.predict(X_train)
            acc_train = getAccuracy(np.argmax(y_train, 1), predictions_train)

            print(f"[{epoch + 1}/{self.epochNo}] ------> Training :  Accuracy : {acc_train}")
            print(f"[{epoch + 1}/{self.epochNo}] ------> Training :  Loss     : {cross_loss_train}")
            print('______________________________________________________________________________________\n')

    def predict(self, X):
        _, _, A2 = forwardPass(X, self.weights, self.layerSizes)
        y_predIndex = np.argmax(A2[149], axis=1)
        return y_predIndex
