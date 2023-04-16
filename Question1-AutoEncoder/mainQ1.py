import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import ImagePreprocessing as IP
import AutoEncoder as AE


def read_data(filePath):
    # Read the data with its path location
    try:
        file = h5py.File(filePath, 'r')
        print('File path is correct!!!')
        return file
    except Exception:
        print('Invalid file path!!!')
        sys.exit(1)


file = read_data(filePath='../Datasets/data1.h5')
keys = list(file.keys())
data = file[keys[0]][()]
invXForm = file[keys[1]][()]
xForm = file[keys[2]][()]

print('The shape of the image data is:', data.shape)

imPP = IP.ImagePreprocessing(data)
imPP.convertGrayscale()
imageData = imPP.normalize(3, (0.1, 0.9))
print('The minimum value of the normalized dataset is:', imageData.min())
print('The maximum value of the normalized dataset is:', imageData.max())
imPP.displayImages(data, imageData, 200)

shapeData = imageData.shape
imageData = imageData.reshape(shapeData[0], shapeData[1] * shapeData[2])
print('The shape of the flattened image data is:', imageData.shape)

Lhid = 64
lambdaVal = 5e-4
betaList = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
rhoList = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
epochNo = 3000
learningRate = 7e-1

J_list = []
We = AE.initializeWeights(imageData.shape[1], Lhid)
for beta in betaList:
    tempJ_list = []
    for rho in rhoList:
        parameters = {'lambda': lambdaVal,
                      'beta': beta,
                      'rho': rho}
        J, J_grad = AE.aeCost(We, imageData, parameters)
        tempJ_list.append(J)
    J_list.append(tempJ_list)
minIndex = np.argwhere(J_list == np.min(J_list)).ravel()

bestBeta = betaList[minIndex[0]]
bestRho = rhoList[minIndex[1]]
print('Chosen beta value is:', bestBeta)
print('Chosen rho value is:', bestRho)
parameters = {'lambda': lambdaVal,
              'beta': bestBeta,
              'rho': bestRho}

modelAE = AE.AutoEncoder(We, parameters, epochNo, learningRate)
We, J_list = modelAE.solver(imageData)
plt.figure()
plt.title('Loss Value at Each Epoch')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(J_list)
modelAE.displayHiddenWeights('Hidden Layer Features', 10)
modelAE.reconstruct(imageData, imageNumber=15)

LhidList = [10, 50, 100]
lambdaValList = [1e-5, 1e-3, 0]
epochNo = 1000

for hidSize in LhidList:
    for lam in lambdaValList:
        We = AE.initializeWeights(imageData.shape[1], hidSize)
        parameters = {'lambda': lam,
                      'beta': bestBeta,
                      'rho': bestRho}
        modelAE = AE.AutoEncoder(We, parameters, epochNo, learningRate)
        We, J_list = modelAE.solver(imageData)
        title = 'Hidden Layer Features for Lhid = ' + str(hidSize) + ' and for lambda = ' + str(lam)
        modelAE.displayHiddenWeights(title, 10)


