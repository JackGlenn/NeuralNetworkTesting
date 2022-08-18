import string
import struct
from matplotlib import image
import numpy as np
from PIL import Image
import random

import matplotlib.pyplot as plt


def generateDataArray(mnistFilePath: string) -> np.ndarray:
    with open(mnistFilePath, 'rb') as file:
        # read first 4 bytes, tossing out unused first two bytes of magic number when unpacking
        magic = struct.unpack('>xxbb',file.read(4))
        numDim = magic[1]
        # dimension sizes are 4 byte unsigned ints
        # number of Images, Rows, Col
        numImg, numRow, numCol = struct.unpack('>' + 'I'*numDim, file.read(12))

        # total number of data bytes to be read
        numBytes = numImg * numRow * numCol
        # value bound: [0, 255]
        dataArray = np.asarray(struct.unpack('>' + 'B'*numBytes, file.read(numBytes)), dtype=np.uint8).reshape((numImg, numRow, numCol))
    return dataArray

def generateLabelArray(labelFilePath: string) -> np.ndarray:
    with open(labelFilePath, 'rb') as file:
        magic = struct.unpack('>xxbb', file.read(4))
        numDim = magic[1]
        numLabels = struct.unpack('>' + 'I' * numDim, file.read(4))[0]
        labelArray = np.asarray(struct.unpack('>' + 'B'*numLabels, file.read(numLabels)), dtype=np.uint8)
    return labelArray

def printImage(imageArray):
    for i in range(28):
        for j in range(28):
            if imageArray[i][j] == 0:
                print(' ',end='')
            else:
                print('1', end='')
        print()
    print(flush=True)

imageArray = generateDataArray("MNIST/t10k-images.idx3-ubyte")
# trainArray = generateDataArray("MNIST/train-images.idx3-ubyte")
labelArray = generateLabelArray("MNIST/t10k-labels.idx1-ubyte")

while True:
    trash = input()
    if trash == "quit":
        break
    index = random.randrange(len(imageArray))
    print(labelArray[index])
    plt.imshow(imageArray[index])
    plt.show()
# printImage(imageArray[0])
# net = NeuralNetwork([784, 30, 10])
# print(labelArray[0])