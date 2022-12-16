import numpy as np


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


def forwardPass(X, weights, layerSizes):
    W1 = weights['W1']
    WR = weights['WR']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']

    input_timeStep = dict()
    hidden_timeStep = dict()
    output_timeStep = dict()

    A1_0 = np.zeros((1, layerSizes['hiddenSize']))
    hidden_timeStep[-1] = np.copy(A1_0)
    for timeStep in range(layerSizes['timeStepSize']):
        currentTimeSample = X[:, timeStep, :]
        input_timeStep[timeStep] = currentTimeSample

        current_Z1 = np.dot(currentTimeSample, W1) + np.dot(hidden_timeStep[timeStep-1], WR) + b1
        current_A1 = tanh(current_Z1, 'forward')
        hidden_timeStep[timeStep] = current_A1

        current_Z2 = np.dot(hidden_timeStep[timeStep], W2) + b2
        current_A2 = softmax(current_Z2)
        output_timeStep[timeStep] = current_A2

    return input_timeStep, hidden_timeStep, output_timeStep


def backwardPass(y, cache, weights, layerSizes):
    input_timeStep, hidden_timeStep, output_timeStep = cache

    sampleSize = y.shape[0]
    timeStepSize = layerSizes['timeStepSize']
    featureSize = layerSizes['featureSize']
    hiddenSize = layerSizes['hiddenSize']
    outputSize = layerSizes['outputSize']

    WR = weights['WR']
    W2 = weights['W2']

    dW1 = np.zeros((featureSize, hiddenSize))
    dWR = np.zeros((hiddenSize, hiddenSize))
    dW2 = np.zeros((hiddenSize, outputSize))

    db1 = np.zeros((1, hiddenSize))
    db2 = np.zeros((1, outputSize))
    dA1_prev = np.zeros((sampleSize, hiddenSize))

    dA2 = np.copy(output_timeStep[149])
    dA2[np.arange(len(y)), np.argmax(y, axis=1)] -= 1

    dW2 += np.dot(hidden_timeStep[149].T, dA2)
    db2 += np.sum(dA2, axis=0, keepdims=True)

    for reversedTimeStep in reversed(range(1, timeStepSize)):
        dA1 = np.dot(dA2, W2.T) + dA1_prev
        dZ1 = tanh(hidden_timeStep[reversedTimeStep], 'gradient') * dA1

        dW1 += np.dot(input_timeStep[reversedTimeStep].T, dZ1)
        dWR += np.dot(hidden_timeStep[reversedTimeStep-1].T, dZ1)
        db1 += np.sum(dZ1, axis=0, keepdims=True)

        dA1_prev = np.dot(dZ1, WR.T)

    for weightDerivatives in [dW1, db1, dWR, dW2, db2]:
        np.clip(weightDerivatives, -10, 10, out=weightDerivatives)

    J_grad = {'dW2': dW2,
              'db2': db2,
              'dW1': dW1,
              'dWR': dWR,
              'db1': db1}
    return J_grad


def crossEntropyLoss(y_true, y_pred):
    m = y_pred.shape[0]
    lossCE = -np.sum(y_true * np.log(y_pred)) / m
    return lossCE


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


def accuracy(labels, preds):
    return 100 * (labels == preds).mean()

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

        self.train_loss, self.test_loss, self.train_acc, self.test_acc = [], [], [], []


    def fit(self, X_train, y_train, X_val, y_val):
        np.random.seed(150)
        self.weights = initializeWeights(self.layerSizes)

        m = X_train.shape[0]
        if m % self.batchSize == 0:
            iterationNo = m // self.batchSize
        else:
            iterationNo = m // self.batchSize + 1

        for epoch in range(self.epochNo):
            print(f'Epoch : {epoch + 1}')
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
            cross_loss_train = crossEntropyLoss(y_batch, cache[2][149])
            predictions_train = self.predict(X_train)
            acc_train = accuracy(np.argmax(y_train, 1), predictions_train)

            _, __, probs_test = forwardPass(X_val, self.weights, self.layerSizes)
            cross_loss_val = crossEntropyLoss(y_val, probs_test[149])
            predictions_val = np.argmax(probs_test[149], 1)
            acc_val = accuracy(np.argmax(y_val, 1), predictions_val)

            print(f"[{epoch + 1}/{self.epochNo}] ------> Training :  Accuracy : {acc_train}")
            print(f"[{epoch + 1}/{self.epochNo}] ------> Training :  Loss     : {cross_loss_train}")
            print('______________________________________________________________________________________\n')
            print(f"[{epoch + 1}/{self.epochNo}] ------> Testing  :  Accuracy : {acc_val}")
            print(f"[{epoch + 1}/{self.epochNo}] ------> Testing  :  Loss     : {cross_loss_val}")
            print('______________________________________________________________________________________\n')

            self.train_loss.append(cross_loss_train)
            self.test_loss.append(cross_loss_val)
            self.train_acc.append(acc_train)
            self.test_acc.append(acc_val)

    def predict(self, X):
        _, __, y_pred = forwardPass(X, self.weights, self.layerSizes)
        y_predIndex = np.argmax(y_pred[149], axis=1)
        return y_predIndex