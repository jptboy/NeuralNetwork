from tensorflow.keras.datasets import mnist
train, test = mnist.load_data()
inputs, targets = train
import numpy as np
trainx, trainy = test

for i in range(len(inputs)):
    inputs[i] = np.flatten(inputs[i])
for i in range(len(trainx)):
    trainx[i] = np.flatten(trainx[i])
import numpy as np


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
    LinearLayer(inputSize=16, outputSize=10),
    Softmax()
])

train(net, inputs, targets, loss= CrossEntropy(), num_epochs=5000, optimizer=MBGD(learningRate=0.01))

# net.loadParamsFromFile("/home/ayush/scratch/Net/aknet/serializedMNIST.json")

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)

net.serialize("serializedMNIST.json")
