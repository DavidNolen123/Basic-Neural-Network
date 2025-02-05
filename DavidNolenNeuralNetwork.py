#David's Neural Network

#My lovely imports

#tensor data --- cite??
#import tensorflow as tensorflow
#from  tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data/', one_hot=True) no no no

import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Download the MNIST dataset (trainImage = list of images, image[x] would return an array of grey scale pixels, trainLabel = image labels, e.g. 0,3,4


(trainImage, trainLabel), (testImage, testLabel) = mnist.load_data()


#making it floaty
trainImage = trainImage.astype(float)

testImage = testImage.astype(float)

print("image array with grescales")
print(trainImage[0])
print("should return that line")
print(trainImage[0][0])
print("should return first greyscale val")
print(trainImage[0][0][0])
print("should reutr label")
print(trainLabel[0])

#28 * 28 = 784 (pixel per img)

import tkinter as tk
from tkinter import *

#https://stackoverflow.com/questions/9506841/using-pil-to-turn-a-rgb-image-into-a-pure-black-and-white-image
#this is  how to    put itt into black gand whitke

#kinter stuffs

#prints image1 out
canvasSetup = Tk()
canvasSetup.title('Image in Tkinter (simple GUI)')
canvasSetup.geometry('800x400')

#canvasSetup1 = Tk()
#canvasSetup1.title('MNIST IMAGE)')
#canvasSetup1.geometry('800x400')

#i forget what PIL is!
from PIL import Image

image1 = Image.open(r'C:\Users\david\OneDrive\Pictures\Screenshots\Screenshot 2024-06-13 164036.png')
sizeLevel = 400

#initial size calculadora (for the image image display
thePixels = list(image1.getdata())
image1Size = image1.size
pixelLength1 = image1Size[0]
pixelWidth1 = image1Size[1]

ratio = pixelLength1/pixelWidth1#such that new image matches scale of orgin

newSize = (int(ratio*sizeLevel), sizeLevel)#""
image1 = image1.resize(newSize)#such that it loads fasta
#image1.show()

thePixels = list(image1.getdata())
image1Size = image1.size #a tuple is like a list but cantnot be modfied

pixelLength1 = image1Size[0]
pixelWidth1 = image1Size[1]

print(thePixels[0])

print(pixelLength1)
print(pixelWidth1)
print(image1Size)


def convertRBGtoHEX(red, green, blue):
    
    return '#%02x%02x%02x' % (red, green, blue)



#image canvas
lineRangeVal = pixelLength1 * pixelWidth1

thisCanvas = Canvas(canvasSetup, width=pixelLength1, height=pixelWidth1, bg='white')

thisCanvas.pack(pady=20)



#thisCanvas1 = Canvas(canvasSetup, width=pixelLength1, height=pixelWidth1, bg='white')

#thisCanvas1.pack(pady=20)




#pixelArrangementCanvas

#thisCanvas2 = Canvas(canvasSetup, width=pixelLength1, height=pixelWidth1, bg='white')

#thisCanvas2.pack(pady=20)

def printMyMNIST(mnistIMG, offsetX, offsetY):

    offsetX *= 28
    offsetY *= 28
    
    for i in range(784):

      rowYmn = int(i/28)
      rowXmn = i % 28

      aPixelVal = int(trainImage[mnistIMG][rowYmn][rowXmn])#i-1 really
        #image number, array line number, value/column number

      GreyRBG = [aPixelVal, aPixelVal, aPixelVal]#appearently this will be grey....

      thisCanvas.create_line(rowXmn + offsetX,rowYmn + offsetY,rowXmn + offsetX ,rowYmn + offsetY + 1,fill=convertRBGtoHEX(GreyRBG[0],GreyRBG[1],GreyRBG[2]))


      #that was used to display img
      #thisCanvas.create_line(rowX,rowY,rowX,rowY+1,fill=convertRBGtoHEX(thePixels[i-1][0],thePixels[i-1][1],thePixels[i-1][2]))
      #print(int(i/pixelLength1))
      
      if(i==lineRangeVal-10):
          print('printed')

for i in range(250):

    rowYmn = int(i/17)
    rowXmn = i % 17
    
    printMyMNIST(i, rowXmn, rowYmn)



trainingRange = len(trainImage)#60,000
print(trainingRange)

testingRange = len(testImage)#10,000
print(testingRange)

pixelList = []


#I will have two hidden layers for now
#784 by whatever amount of hidden layer neurons
#weight matrixes
minput_2weights = []
m2_3weights = []
m3_finalweights = []

#will be multiplying inputToFrist... by the 784*1 matrix to get a [20 x 1]
mi_2wROW = 20
mi_2wCOLUMN = 784

m2_3wROW = 20
m2_3wCOLUMN = 20

m3_finalwROW = 10
m3_finalwCOLUMN = 20

def randomizeMatrix (matrixRow, matrixColumn, finalMatrix, minim, maxim, decis):

    #finalMatrix = []   #commented out, but make sure that matrix is blank or there WILL be a dimension mismatch in proceeding calculations
    
    for theRow in range(matrixRow):
        z = []
    
        for theColumn in range(matrixColumn):   
            z.append(round(random.uniform(minim,maxim),decis))
        finalMatrix.append(z)

randomizeMatrix(mi_2wROW, mi_2wCOLUMN, minput_2weights, -0.5, 0.5, 2)
randomizeMatrix(m2_3wROW, m2_3wCOLUMN, m2_3weights, -0.5, 0.5, 2)
randomizeMatrix(m3_finalwROW, m3_finalwCOLUMN, m3_finalweights, -0.5, 0.5, 2)

#prints out all the random weights
print(minput_2weights)
print(m2_3weights)
print(m3_finalweights)

#@ symbol for matrix mult

inputLayer = []

inputLayerROW = 784
inputLayerCOLUMN = 1


hiddenLayer1 = []
hiddenLayer2 = []
outputLayer = []

#for now, just one image

e = 2.71828 #euler's constant

#yay sigmoid funtioono
def sigmoidify(myNumber):
    #return 1 / (1 + (e**(-myNumber)))
    return 1 / (1 + np.exp(-myNumber))# a more precise e


#this exmp only wrks with img 0

        
for i in range(784):

    rowYmn = int(i/28)
    rowXmn = i % 28

    #simply normalizes numbers
    replacement = ((trainImage[0][rowYmn][rowXmn]))/float(255.0)
    #print(replacement)
    
    trainImage[0][rowYmn][rowXmn] = replacement #uhhhh

    """sigmoidify"""
    #I guess this didnt' give null multiplyer error bc it was not a funtion therefore no "return" necessary
    
    #but this is still a 28*28 array matrix...

      
#so I have as follows:
#turns it into a 784*1
for theRow in range(inputLayerROW):#and that is..784

        z = []

        rowYmn = int(theRow/28)
        rowXmn = theRow % 28

        for theColumn in range(inputLayerCOLUMN):
            
            z.append(trainImage[0][rowYmn][rowXmn])
            
        inputLayer.append(z)




def sigmoidifyMatrix(matrixRow, matrixColumn, matrix):
    for i in range(matrixRow * matrixColumn):

        row = int(i/matrixColumn)
        column = i % matrixColumn#not backwards actually, 27%28 = 27

        #simply normalizes numbers
        replacement = matrix[row][column]
        #print(replacement)
        
        matrix[row][column] = sigmoidify(replacement)

    return matrix
    #aha maybe this will finally fix the "null type" error in the dot product


hiddenLayer1 = np.dot(minput_2weights, inputLayer)
hiddenLayer1 = sigmoidifyMatrix(len(hiddenLayer1), len(hiddenLayer1[0]), hiddenLayer1)

hiddenLayer2 = np.dot(m2_3weights, hiddenLayer1)
hiddenLayer2 = sigmoidifyMatrix(len(hiddenLayer2), len(hiddenLayer2[0]), hiddenLayer2)

outputLayer = np.dot(m3_finalweights, hiddenLayer2)
outputLayer = sigmoidifyMatrix(len(outputLayer), len(outputLayer[0]), outputLayer)

print("input layer stats")
print("rows", len(inputLayer))
print("columns", len(inputLayer[0]))
print(inputLayer)

#print("item number..1", inputLayer[0][0])

print("hidden layer 1 stats:")
print("rows", len(hiddenLayer1))
print("columns", len(hiddenLayer1[0]))
print(hiddenLayer1)

#print("item number..1", hiddenLayer1[0][0])'''

print("hidden layer 2 stats:")
print("rows", len(hiddenLayer2))
print("columns", len(hiddenLayer2[0]))
print(hiddenLayer2)



print("output layer stats:")
print("rows", len(outputLayer))
print("columns", len(outputLayer[0]))
print(outputLayer)

#please please work please

"""print(hiddenLayer1)
print(hiddenLayer2)"""
#print(outputLayer)

