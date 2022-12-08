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
imPP.displayImages()

shapeData = imageData.shape
imageData = imageData.reshape(shapeData[0], shapeData[1] * shapeData[2])
print('The shape of the flattened image data is:', imageData.shape)

Lhid = 64
lambdaVal = 5 * (10 ** -4)
beta = 0.01
rho = 0.1

parameters = {'Lin': imageData.shape[1],
              'Lhid': Lhid,
              'lambda': lambdaVal,
              'beta': beta,
              'rho': rho}

We = AE.fit(imageData, parameters, 100, 0.1)
AE.predict(We, imageData)
