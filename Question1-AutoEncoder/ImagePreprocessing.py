import matplotlib.pyplot as plt
import numpy as np


class ImagePreprocessing:
    def __init__(self, imageData):
        self.imageData = imageData
        self.grayscaleData = None

    def convertGrayscale(self):
        rLuminosity = self.imageData[:, 0, :, :] * 0.2126
        gLuminosity = self.imageData[:, 1, :, :] * 0.7152
        bLuminosity = self.imageData[:, 2, :, :] * 0.0722
        self.grayscaleData = rLuminosity + gLuminosity + bLuminosity
        print('The shape of the grayscale image data is:', self.grayscaleData.shape)

    def removeMean(self):
        meanData = self.grayscaleData.mean(axis=(1, 2), keepdims=True)
        self.grayscaleData -= meanData

    def clipData(self, clipParameter):
        stdData = np.std(self.grayscaleData)
        minVal = -clipParameter * stdData
        maxVal = clipParameter * stdData
        self.grayscaleData = np.clip(self.grayscaleData, a_min=minVal, a_max=maxVal)

    def normalize(self, clipParameter, normalizeRange):
        self.removeMean()
        self.clipData(clipParameter)
        # normalize between 0 and 1
        normalizedData = (self.grayscaleData - self.grayscaleData.min())
        normalizedData /= (self.grayscaleData.max() - self.grayscaleData.min())
        # normalize between 0.1 and 0.9
        normalizedData = normalizedData * (normalizeRange[1]-normalizeRange[0]) + normalizeRange[0]
        return normalizedData

    def displayImages(self):
        # TODO create display function
        pass
