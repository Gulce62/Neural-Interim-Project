import numpy as np
import matplotlib.pyplot as plt


def initialization(Lpre, Lpost, size):
    np.random.seed(0)  # TODO look for different seed values
    wo = np.sqrt(6 / (Lpre + Lpost))
    parameter = np.random.uniform(-wo, wo, size=size)
    return parameter


def initializeWeights(inputSize, hiddenSize):
    W1 = initialization(inputSize, hiddenSize, (inputSize, hiddenSize))
    W2 = W1.T
    b1 = initialization(inputSize, hiddenSize, (1, hiddenSize))
    b2 = initialization(inputSize, hiddenSize, (1, inputSize))
    We = {'W1': W1,
          'W2': W2,
          'b1': b1,
          'b2': b2}
    return We


def sigmoid(z, condition):
    s = 1 / (1 + np.exp(-z))
    if condition == 'forward':
        return s
    if condition == 'gradient':
        return s * (1 - s)


def forwardPass(We, data):
    W1 = We['W1']
    b1 = We['b1']
    W2 = We['W2']
    b2 = We['b2']

    Z1 = np.dot(data, W1) + b1
    A1 = sigmoid(Z1, 'forward')
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2, 'forward')

    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}
    return cache


def MSE(oDesired, oNetwork, condition):
    if condition == 'loss':
        m = oDesired.shape[0]
        lossMSE = np.sum(np.square(oDesired - oNetwork)) / (2 * m)
        return lossMSE
    if condition == 'gradient':
        gradMSE = (1 / 2) * (oNetwork - oDesired)
        return gradMSE


def TykhonovRegularization(lambdaVal, We, condition):
    W1 = We['W1']
    W2 = We['W2']
    if condition == 'loss':
        regW1 = np.sum(np.square(W1))
        regW2 = np.sum(np.square(W2))
        lossTR = (lambdaVal / 2) * (regW1 + regW2)
        return lossTR
    if condition == 'gradient':
        gradW1 = lambdaVal * W1
        gradW2 = lambdaVal * W2
        return gradW1, gradW2


def KL(beta, rho, A1, condition):
    rho_b = A1.mean(axis=0, keepdims=True)
    if condition == 'loss':
        lossKL1 = rho * np.log(rho / rho_b)
        lossKL2 = (1 - rho) * np.log((1 - rho) / (1 - rho_b))
        lossKL = beta * np.sum(lossKL1 + lossKL2)
        return lossKL
    if condition == 'gradient':
        gradKL1 = -rho / rho_b
        gradKL2 = (1 - rho) / (1 - rho_b)
        gradKL = beta * (gradKL1 + gradKL2) * (1 / A1.shape[1])
        return gradKL


def backwardPass(We, data, params, cache):
    m = data.shape[0]

    lambdaVal = params['lambda']
    beta = params['beta']
    rho = params['rho']

    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']

    dTRW1, dTRW2 = TykhonovRegularization(lambdaVal, We, 'gradient')
    dKL = KL(beta, rho, A1, 'gradient')

    dZ2 = MSE(data, A2, 'gradient') * sigmoid(Z2, 'gradient')
    dW2 = (1 / m) * (np.dot(A1.T, dZ2) + dTRW2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = (np.dot(dZ2, We['W2'].T) + dKL) * sigmoid(Z1, 'gradient')
    dW1 = (1 / m) * (np.dot(data.T, dZ1) + dTRW1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    grads = {'dW1': dW1,
             'db1': db1,
             'dW2': dW2,
             'db2': db2}
    return grads


def aeCost(We, data, params):
    # forward pass
    cache = forwardPass(We, data)

    # calculate loss
    lossMSE = MSE(data, cache['A2'], 'loss')
    lossTR = TykhonovRegularization(params['lambda'], We, 'loss')
    lossKL = KL(params['beta'], params['rho'], cache['A1'], 'loss')
    J = lossMSE + lossTR + lossKL

    # backward pass
    J_grad = backwardPass(We, data, params, cache)

    return J, J_grad


def updateParameters(We, J_grad, learningRate):
    We['W1'] -= learningRate * J_grad['dW1']
    We['b1'] -= learningRate * J_grad['db1']
    We['W2'] -= learningRate * J_grad['dW2']
    We['b2'] -= learningRate * J_grad['db2']
    return We


class AutoEncoder:
    def __init__(self, We, parameters, epochNo, learningRate):
        self.We = We
        self.parameters = parameters
        self.epochNo = epochNo
        self.learningRate = learningRate

    def solver(self, data):
        J_list = []
        for epoch in range(self.epochNo):
            J, J_grad = aeCost(self.We, data, self.parameters)
            self.We = updateParameters(self.We, J_grad, self.learningRate)
            # self.learningRate *= 0.9999
            J_list.append(J)
            print('The cost at epoch =', str(epoch), 'is:', str(J))
        return self.We, J_list

    def displayHiddenWeights(self):
        W1 = self.We['W1']
        plt.figure(figsize=(9, 8))
        plt.suptitle('Hidden Layer Features')
        for w in range(W1.shape[1]):
            plt.subplot(8, 8, w + 1)
            plt.imshow(W1.T[w].reshape(16, 16), cmap='gray')
            plt.axis('off')
        plt.show()

    def reconstruct(self, data, imageNumber):
        cache = forwardPass(self.We, data)
        reconstructedImages = cache['A2']

        np.random.seed(0)
        indices = np.random.randint(data.shape[0], size=imageNumber)

        plt.figure(figsize=(16, 12))
        for image in range(imageNumber):
            if imageNumber % 5 == 0:
                pos = imageNumber // 5
            else:
                pos = imageNumber // 5 + 1
            plt.subplot(2 * pos, 5, image + 1)
            plt.title("Reconstructed Image")
            plt.imshow(reconstructedImages[indices[image]].reshape(16, 16), cmap='gray')
            plt.axis('off')

            plt.subplot(2 * pos, 5, image + 1 + (5 * pos))
            plt.title("Original Image")
            plt.imshow(data[indices[image]].reshape(16, 16), cmap='gray')
            plt.axis('off')
        plt.show()
