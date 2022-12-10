import matplotlib.pyplot as plt
import numpy as np


class ImagePreprocessing:
    def __init__(self, imageData):
        self.imageData = imageData
        self.grayscaleData = None

    def convertGrayscale(self):
        # Convert images to grayscale using a luminosity model
        rLuminosity = self.imageData[:, 0, :, :] * 0.2126  # red
        gLuminosity = self.imageData[:, 1, :, :] * 0.7152  # green
        bLuminosity = self.imageData[:, 2, :, :] * 0.0722  # blue
        self.grayscaleData = rLuminosity + gLuminosity + bLuminosity
        print('The shape of the grayscale image data is:', self.grayscaleData.shape)

    def removeMean(self):
        # Remove the mean pixel intensity of each image
        meanData = self.grayscaleData.mean(axis=(1, 2), keepdims=True)
        self.grayscaleData -= meanData

    def clipData(self, clipParameter):
        # Clip the data range at +-3 standard deviations
        stdData = np.std(self.grayscaleData)
        minVal = -clipParameter * stdData
        maxVal = clipParameter * stdData
        self.grayscaleData = np.clip(self.grayscaleData, a_min=minVal, a_max=maxVal)

    def normalize(self, clipParameter, normalizeRange):
        self.removeMean()
        self.clipData(clipParameter)
        # Normalize data between 0 and 1
        normalizedData = (self.grayscaleData - self.grayscaleData.min())
        normalizedData /= (self.grayscaleData.max() - self.grayscaleData.min())
        # Normalize data between 0.1 and 0.9
        normalizedData = normalizedData * (normalizeRange[1]-normalizeRange[0]) + normalizeRange[0]
        return normalizedData

    def displayImages(self, data, imageData, imageNumber):
        # Display random image patches with desired image number
        np.random.seed(0)
        indices = np.random.randint(data.shape[0], size=imageNumber) # Find random indices

        plt.figure(figsize=(16, 16))
        plt.suptitle('The ' + str(imageNumber) + ' random sample patches in RGB and normalized grayscale format', fontsize=25)

        # Display the data (matplotlib automatically clip the data)
        data = np.moveaxis(data, source=1, destination=3)
        for image in range(imageNumber):
            plt.subplot(32, 20, image + 1)
            plt.imshow(data[indices[image]])
            plt.axis('off')

        # Display normalized data between 0 and 1
        data = (data-data.min())/(data.max()-data.min())
        for image in range(imageNumber):
            plt.subplot(32, 20, image + imageNumber + 21)
            plt.imshow(data[indices[image]])
            plt.axis('off')

        # Display preprocessed data (subtract mean + clip + normalize)
        for image in range(imageNumber):
            plt.subplot(32, 20, image + 2*imageNumber + 41)
            plt.imshow(imageData[indices[image]], cmap='gray')
            plt.axis('off')
        plt.show()

