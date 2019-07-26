from tensorflow.keras.datasets import mnist
dataMnist = mnist.load_data() 
train, test = dataMnist
inputs, targets = train
import numpy as np
testx, testy = test

import numpy as np
inputs1 = []
trainx1 = []
targets1 = []
for i in range(len(inputs)):
    inputs1.append(inputs[i].flatten())
for i in range(len(testx)):
    trainx1.append(testx[i].flatten())
for i in range(len(targets)):
    foo = np.zeros(10)
    foo[targets[i]] = 1
    targets1.append(foo)

inputs = np.array(inputs1)
testx = np.array(trainx1)
targets = np.array(targets1)

from train import train
from network import NeuralNetwork
from layer import LinearLayer, Sigmoid, Relu, Softmax, Tanh
from loss import MSE, CrossEntropy
from optimizer import MBGD

net = NeuralNetwork([
    LinearLayer(inputSize=784, outputSize=16),
    Tanh(),
    LinearLayer(inputSize=16, outputSize=16),
    Tanh(),
    LinearLayer(inputSize=16, outputSize=16),
    Tanh(),
    LinearLayer(inputSize=16, outputSize=10),
    Softmax()
])

train(net, inputs, targets, loss= CrossEntropy(), num_epochs=80, optimizer=MBGD(learningRate=0.01), showGraph=True)

# net.loadParamsFromFile("/home/ayush/scratch/Net/aknet/serializedMNIST.json")

import matplotlib.pyplot as plt
for x, y in zip(testx, testy):
    predicted = net.forward(x)
    plt.imshow(x.reshape((28,28)))
    plt.show()
    print(np.argmax(predicted), y)

net.serialize("serializedMNIST.json")
