#David's Neural Network

import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist

#import pandas as 
# Download the MNIST dataset (trainImage = list of images, image[x] would return an array of grey scale pixels, trainLabel = image labels, e.g. 0,3,4


(trainImage, trainLabel), (testImage, testLabel) = mnist.load_data()


#making it floaty
trainImage = trainImage.astype(float)

testImage = testImage.astype(float)


'''print("image array with grescales")
print(trainImage[0])
print("should return that line")
print(trainImage[0][0])
print("should return first greyscale val")
print(trainImage[0][0][0])
print("should return label")
print(trainLabel[0])
'''

print("Train image shape:", (trainImage.shape, trainLabel.shape))
print("Test image shape:", (testImage.shape, testLabel.shape))

def one_hot(oneHotInput):
    for i in oneHotInput:
        x.append
    

def loadTheData():
    
    #load data, prepare data, define model, eval, display results
    #already loaded
    #reshape data - why?

    #trainImage = trainImage.reshape((trainImage.shape[0], 28, 28, 1))#it is not clear to me what this does
    #testImage = testImage.reshape((testImage.shape[0], 28, 28, 1))#likewise
    #single color channel

    #one hot encoding = binary
    #trainLabel = one_hot(trainLabel)
    #testLabel = one_hot(testLabel)

    
    #testImage = to_categorical(testImage)
    #STILL NEED TO ADD THE ABOVE FOR ONE HOT ENCODING


    #PRINTING THE FIRST 5 FROM THE TRAIN IMAGES
    #math matplotlib plot
    import matplotlib as matplotlib
    from matplotlib import pyplot as plt
    for i in range(5):
        #setup subplot?
        plt.subplot(330 + 1 + i)
        #pixels in plot
        plt.imshow(trainImage[i], cmap=plt.get_cmap('gray'))
    #display plot

    plt.show()
    return trainImage, trainLabel, testImage, testLabel


def mapPixels(trainI, testI):
    #reducing the 0-255 to 0-1 floating point
    mapTrain = trainI.astype('FL32')
    mapTest = testI.astype('FL32')
    #put a 0 as 0, 255 as 1
    mapTrain = mapTrain / 255.0
    mapTest = mapTest / 255.0

    return train_norm, test_norm

#def applyConvolutionalLayer1(dataSet):
    #take first huge layer, condince to a 1*10 -or 10*1?

    #HERE: [1*784] * [784 * 10] = [1*10] -- input layer, layer 0

    #1*10   * 10*10 -- layer 1
    #1*10   * 10*10 -- layer 2
    #then use onehot to interpret -- output layer



print(trainImage[0])
#loadTheData()
#print(trainImage[0])

myA = np.array(trainImage[0])

flat = myA.reshape(-1)#results in a 1*784

print(flat)
print(flat.shape)


#input layer should already be defined at this point, and there are no weights and biasses, it is just starting as the final

firstLayerWeights = np.zeros((784, 10))# np.array.array([[0], [0]])
firstLayerBiases = np.zeros((1, 10))
firstLayerResult = np.zeros((1, 10))


secondLayerWeights = []
secondLayerBiases = []
secondLayerResult = []


outputLayerWeights = []
outputLayerBiases = []#are there biasses in the final (output layer)?
outputLayerResult = []


def initializeFirstLayerWeights():
    firstLayerWeights = np.random.rand(784,10)
    firstLayerWeights = np.around(firstLayerWeights, decimals = 3)
    return firstLayerWeights

def initializeFirstLayerBiases():
    firstLayerBiases = np.random.rand(10,1)
    firstLayerBiases = np.around(firstLayerWeights, decimals = 3)
    return firstLayerBiases


def initializeSecondLayerWeights():
    secondLayerWeights = np.random.rand(10,10)
    secondLayerWeights = np.around(secondLayerWeights, decimals = 3)
    return secondLayerWeights



firstLayerWeights = initializeFirstLayerWeights()
#initializeFirstLayerBiases()
#print(firstLayerWeights)

def sigMoidMatrix(m):
    m = 1/(1 + np.exp(-m))
    return(m)

inputLayer = trainImage[0].reshape(1, 784)
inputLayer = inputLayer - (255/2)
inputLayer = sigMoidMatrix(inputLayer)

print("inputLayer adjusted", inputLayer)
    
def runFirstLayerProcess():
    secondLayer = np.dot(inputLayer, firstLayerWeights) + firstLayerBiases
    #secondLayer = sigMoidMatrix(secondLayer)
    return(secondLayer)
                    
def runSecondLayerProcess(firstLayerResult, secondLayerWeights, secondLayerBiases):
    x = np.dot(inputLayer, firstLayerWeights) + firstLayerBiases

print(runFirstLayerProcess())

#3blue1brown: a bias is basically just adding one number to the weighted sum
#maybe check out his linear algrebra course?


print("test");
