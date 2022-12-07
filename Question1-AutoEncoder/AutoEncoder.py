import numpy as np


def initialization(Lpre, Lpost):
    np.random.seed(50)  # TODO look for different seed values
    wo = np.sqrt(6 / (Lpre + Lpost))
    parameter = np.random.uniform(-wo, wo, size=(Lpre, Lpost))
    return parameter


def initializeWeights(inputSize, hiddenSize):
    W1 = initialization(inputSize, hiddenSize)
    W2 = W1.T
    b1 = initialization(1, hiddenSize)
    b2 = initialization(1, inputSize)
    We = {'W1': W1,
          'W2': W2,
          'b1': b1,
          'b2': b2}
    return We


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoidGradient(s):
    return s * (1-s)


def forwardPass(We, data):
    W1 = We['W1']
    b1 = We['b1']
    W2 = We['W2']
    b2 = We['b2']

    Z1 = np.dot(data.T, W1) + b1
    print(Z1.shape)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return A2, A1


def lossMSE(oDesired, oNetwork, m):
    MSE = np.sum(np.square(oDesired - oNetwork)) / (2*m)
    return MSE


def lossTykhonovRegularization():
    pass


def lossKL(beta, rho, A1):
    rho_b = A1.mean(axis=0, keepdims=True)
    KL1 = rho * np.log(rho/rho_b)
    KL2 = (1-rho)
    return KL


def backwardPass():
    pass


def aeCost(We, data, params):
    m = data.shape[0]  # the number of samples in the data

    Lin = params['Lin']
    Lhid = params['Lhid']
    lambdaVal = params['lambda']
    beta = params['beta']
    rho = params['rho']

    A2, A1 = forwardPass(We, data)
    MSE = lossMSE(data, A2.T, m)
    print(MSE)
    KL = lossKL(beta, rho, A1)


class AutoEncoder:
    def __init__(self):
        pass

