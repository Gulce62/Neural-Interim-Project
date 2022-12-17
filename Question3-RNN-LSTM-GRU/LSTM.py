import numpy as np


def initialization(Lpre, Lpost, size):
    wo = np.sqrt(6 / (Lpre + Lpost))
    parameter = np.random.uniform(-wo, wo, size=size)
    return parameter


def initializeWeights(layerSizes):
    featureSize = layerSizes['featureSize']
    hiddenSize = layerSizes['hiddenSize']
    outputSize = layerSizes['outputSize']

    W_forget = initialization(featureSize, hiddenSize, (featureSize + hiddenSize, hiddenSize))
    W_input = initialization(featureSize, hiddenSize, (featureSize + hiddenSize, hiddenSize))
    W_cell = initialization(featureSize, hiddenSize, (featureSize + hiddenSize, hiddenSize))
    W_output = initialization(featureSize, hiddenSize, (featureSize + hiddenSize, hiddenSize))
    b_forget = initialization(featureSize, hiddenSize, (1, hiddenSize))
    b_input = initialization(featureSize, hiddenSize, (1, hiddenSize))
    b_cell = initialization(featureSize, hiddenSize, (1, hiddenSize))
    b_output = initialization(featureSize, hiddenSize, (1, hiddenSize))
    W = initialization(hiddenSize, outputSize, (hiddenSize, outputSize))
    b = initialization(hiddenSize, outputSize, (1, outputSize))

    We = {'W_forget': W_forget,
          'W_input': W_input,
          'W_cell': W_cell,
          'W_output': W_output,
          'b_forget': b_forget,
          'b_input': b_input,
          'b_cell': b_cell,
          'b_output': b_output,
          'W': W,
          'b': b}
    return We


def sigmoid(z, condition):
    if condition == 'forward':
        sig = 1 / (1 + np.exp(-z))
        return sig
    if condition == 'gradient':
        return z * (1 - z)


def tanh(z, condition):
    if condition == 'forward':
        tanh = np.tanh(z)
        return tanh
    if condition == 'gradient':
        tanhGrad = 1 - (z * z)
        return tanhGrad


def softmax(z):
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    sm = exp / np.sum(exp, axis=-1, keepdims=True)
    return sm


def forwardPassCell(currentTimeStep, prevHiddenState, prevCellState, weights):
    W_forget = weights['W_forget']
    W_input = weights['W_input']
    W_cell = weights['W_cell']
    W_output = weights['W_output']
    b_forget = weights['b_forget']
    b_input = weights['b_input']
    b_cell = weights['b_cell']
    b_output = weights['b_output']
    W = weights['W']
    b = weights['b']

    currentInput = np.column_stack([currentTimeStep, prevHiddenState])

    currentForgetGate_Z = np.dot(currentInput, W_forget) + b_forget
    currentForgetGate_A = sigmoid(currentForgetGate_Z, 'forward')

    currentInputGate_Z = np.dot(currentInput, W_input) + b_input
    currentInputGate_A = sigmoid(currentInputGate_Z, 'forward')

    currentCellGate_Z = np.dot(currentInput, W_cell) + b_cell
    currentCellGate_A = tanh(currentCellGate_Z, 'forward')

    currentOutputGate_Z = np.dot(currentInput, W_output) + b_output
    currentOutputGate_A = sigmoid(currentOutputGate_Z, 'forward')

    currentCellState = currentForgetGate_A * prevCellState + currentInputGate_A * currentCellGate_A
    currentHiddenState = currentOutputGate_A * tanh(currentCellState, 'forward')

    current_Z = np.dot(currentHiddenState, W) + b
    current_A = softmax(current_Z)

    cache = {'timeStep': currentTimeStep,
             'input': currentInput,
             'forgetGate': currentForgetGate_A,
             'inputGate': currentInputGate_A,
             'cellGate': currentCellGate_A,
             'outputGate': currentOutputGate_A,
             'cellState': currentCellState,
             'hiddenState': currentHiddenState,
             'Z': current_Z,
             'A': current_A}
    return cache


def forwardPass(X, weights, layerSizes):
    timeStep = {}
    input_timeStep = {}
    forgetGate_timeStep = {}
    inputGate_timeStep = {}
    cellGate_timeStep = {}
    outputGate_timeStep = {}
    cellState_timeStep = {}
    hiddenState_timeStep = {}
    hidden_timeStep = {}
    output_timeStep = {}

    prevCellState = np.zeros((X.shape[0], layerSizes['hiddenSize']))
    prevHiddenState = np.zeros((X.shape[0], layerSizes['hiddenSize']))
    for step in range(layerSizes['timeStepSize']):
        cache = forwardPassCell(X[:, step, :], prevHiddenState, prevCellState, weights)

        timeStep[step] = cache['timeStep']
        input_timeStep[step] = cache['input']
        forgetGate_timeStep[step] = cache['forgetGate']
        inputGate_timeStep[step] = cache['inputGate']
        cellGate_timeStep[step] = cache['cellGate']
        outputGate_timeStep[step] = cache['outputGate']
        cellState_timeStep[step] = cache['cellState']
        hiddenState_timeStep[step] = cache['hiddenState']
        hidden_timeStep[step] = cache['Z']
        output_timeStep[step] = cache['A']

        prevHiddenState = cache['hiddenState']
        prevCellState = cache['cellState']

    return timeStep, input_timeStep, forgetGate_timeStep, inputGate_timeStep, \
           cellGate_timeStep, outputGate_timeStep, cellState_timeStep, \
           hiddenState_timeStep, hidden_timeStep, output_timeStep


def crossEntropyLoss(y_true, y_pred):
    m = y_pred.shape[0]
    lossCE = -np.sum(y_true * np.log(y_pred)) / m
    return lossCE


def backwardPassCell(dhiddenState, dCellState, currentInput, forgetGate, inputGate,
                     cellGate, outputGate, cellState, cellStatePrev, weights):
    dout_A = dhiddenState * np.tanh(cellState)
    dout_Z = dout_A * sigmoid(outputGate, 'gradient')
    dW_output = np.dot(currentInput.T, dout_Z)  # todo sum
    db_output = np.sum(dout_Z, axis=0, keepdims=True)  # todo sum

    dCellState += dhiddenState * outputGate * tanh(cellState, 'gradient')  # todo sum
    dcell_A = dCellState * inputGate
    dcell_Z = dcell_A * tanh(cellGate, 'gradient')
    dW_cell = np.dot(currentInput.T, dcell_Z)  # todo sum
    db_cell = np.sum(dcell_Z, axis=0, keepdims=True)  # todo sum

    dinput_A = dCellState * cellGate
    dinput_Z = sigmoid(inputGate, 'gradient') * dinput_A
    dW_input = np.dot(currentInput.T, dinput_Z)  # todo sum
    db_input = np.sum(dinput_Z, axis=0, keepdims=True)  # todo sum

    dforget_A = dCellState * cellStatePrev
    dforget_Z = sigmoid(forgetGate, 'gradient') * dforget_A
    dW_forget = np.dot(currentInput.T, dforget_Z)  # todo sum
    db_forget = np.sum(dforget_Z, axis=0, keepdims=True)  # todo sum

    dcurrentInput = np.dot(dforget_Z, weights['W_forget'].T) + np.dot(dinput_Z, weights['W_input'].T) + \
                    np.dot(dcell_Z, weights['W_cell'].T) + np.dot(dout_Z, weights['W_output'].T)

    grad = {'dW_output': dW_output,
            'db_output': db_output,
            'dW_cell': dW_cell,
            'db_cell': db_cell,
            'dW_input': dW_input,
            'db_input': db_input,
            'dW_forget': dW_forget,
            'db_forget': db_forget,
            'dcurrentInput': dcurrentInput,
            'dCellState': dCellState}
    return grad


def backwardPass(y, cache, weights, layerSizes):
    timeStep, input_timeStep, forgetGate_timeStep, inputGate_timeStep, \
    cellGate_timeStep, outputGate_timeStep, cellState_timeStep, \
    hiddenState_timeStep, hidden_timeStep, output_timeStep = cache

    sampleSize = y.shape[0]
    timeStepSize = layerSizes['timeStepSize']
    featureSize = layerSizes['featureSize']
    hiddenSize = layerSizes['hiddenSize']
    outputSize = layerSizes['outputSize']

    dW_forget = np.zeros((featureSize + hiddenSize, hiddenSize))
    dW_input = np.zeros((featureSize + hiddenSize, hiddenSize))
    dW_cell = np.zeros((featureSize + hiddenSize, hiddenSize))
    dW_output = np.zeros((featureSize + hiddenSize, hiddenSize))
    db_forget = np.zeros((1, hiddenSize))
    db_input = np.zeros((1, hiddenSize))
    db_cell = np.zeros((1, hiddenSize))
    db_output = np.zeros((1, hiddenSize))
    dW = np.zeros((hiddenSize, outputSize))
    db = np.zeros((1, outputSize))

    dprevCellState = np.zeros((sampleSize, hiddenSize))
    dprevHiddenState = np.zeros((sampleSize, hiddenSize))

    dA = np.copy(output_timeStep[149])
    length = len(y)
    index = np.argmax(y, axis=1)
    dA[np.arange(length), index] -= 1

    dW = np.dot(hiddenState_timeStep[149].T, dA)
    db = np.sum(dA, axis=0, keepdims=True)

    for step in reversed(range(1, timeStepSize)):
        dCellState = np.copy(dprevCellState)
        dHiddenState = np.dot(dA, weights['W'].T) + dprevHiddenState
        grads = backwardPassCell(dHiddenState, dCellState, input_timeStep[step], forgetGate_timeStep[step],
                                 inputGate_timeStep[step], cellGate_timeStep[step], outputGate_timeStep[step],
                                 cellState_timeStep[step], cellState_timeStep[step-1], weights)
        dW_output += grads['dW_output']
        db_output += grads['db_output']
        dW_cell += grads['dW_cell']
        db_cell += grads['db_cell']
        dW_input += grads['dW_input']
        db_input += grads['db_input']
        dW_forget += grads['dW_forget']
        db_forget += grads['db_forget']
        dprevCellState = grads['dcurrentInput'][:,-hiddenSize:]
        dprevHiddenState = forgetGate_timeStep[step] * grads['dCellState']

    weightList = [dW, db, dW_output, db_output, dW_cell, db_cell, db_input, db_input, dW_forget, db_forget]
    for weightDerivatives in weightList:
        np.clip(weightDerivatives, -10, 10, out=weightDerivatives)

    J_grad = {'dW_output': dW_output,
              'db_output': db_output,
              'dW_cell': dW_cell,
              'db_cell': db_cell,
              'dW_input': dW_input,
              'db_input': db_input,
              'dW_forget': dW_forget,
              'db_forget': db_forget,
              'dW': dW,
              'db': db}
    return J_grad


def updateParameters(weights, prevWeights, J_grad, learningRate, momentumRate):
    prevWeights['dW_output_prev'] = learningRate * J_grad['dW_output'] + momentumRate * prevWeights['dW_output_prev']
    prevWeights['db_output_prev'] = learningRate * J_grad['db_output'] + momentumRate * prevWeights['db_output_prev']
    prevWeights['dW_cell_prev'] = learningRate * J_grad['dW_cell'] + momentumRate * prevWeights['dW_cell_prev']
    prevWeights['db_cell_prev'] = learningRate * J_grad['db_cell'] + momentumRate * prevWeights['db_cell_prev']
    prevWeights['dW_input_prev'] = learningRate * J_grad['dW_input'] + momentumRate * prevWeights['dW_input_prev']
    prevWeights['db_input_prev'] = learningRate * J_grad['db_input'] + momentumRate * prevWeights['db_input_prev']
    prevWeights['dW_forget_prev'] = learningRate * J_grad['dW_forget'] + momentumRate * prevWeights['dW_forget_prev']
    prevWeights['db_forget_prev'] = learningRate * J_grad['db_forget'] + momentumRate * prevWeights['db_forget_prev']
    prevWeights['dW_prev'] = learningRate * J_grad['dW'] + momentumRate * prevWeights['dW_prev']
    prevWeights['db_prev'] = learningRate * J_grad['db'] + momentumRate * prevWeights['db_prev']

    weights['W_output'] -= prevWeights['dW_output_prev']
    weights['b_output'] -= prevWeights['db_output_prev']
    weights['W_cell'] -= prevWeights['dW_cell_prev']
    weights['b_cell'] -= prevWeights['db_cell_prev']
    weights['W_input'] -= prevWeights['dW_input_prev']
    weights['b_input'] -= prevWeights['db_input_prev']
    weights['W_forget'] -= prevWeights['dW_forget_prev']
    weights['b_forget'] -= prevWeights['db_forget_prev']
    weights['W'] -= prevWeights['dW_prev']
    weights['b'] -= prevWeights['db_prev']

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
        self.prevWeights = {'dW_output_prev': 0,
                            'db_output_prev': 0,
                            'dW_cell_prev': 0,
                            'db_cell_prev': 0,
                            'dW_input_prev': 0,
                            'db_input_prev': 0,
                            'dW_forget_prev': 0,
                            'db_forget_prev': 0,
                            'dW_prev': 0,
                            'db_prev': 0}

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
                                                                                     J_grad, self.learningRate,
                                                                                     self.momentumRate)
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
        y_pred = cache[9][149]
        y_predIndex = np.argmax(y_pred, axis=1)
        return y_pred, y_predIndex
