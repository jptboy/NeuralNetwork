"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np


from train import train
from network import NeuralNetwork
from layer import LinearLayer, Sigmoid, Relu, Softmax, Tanh
from loss import MSE
from optimizer import MBGD

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNetwork([
    LinearLayer(inputSize=2, outputSize=60),
    Tanh(),
    LinearLayer(inputSize=60, outputSize=60),
    Tanh(),
    LinearLayer(inputSize=60, outputSize=60),
    Tanh(),
    LinearLayer(inputSize=60, outputSize=2)
])

# train(net, inputs, targets, loss= MSE(), num_epochs=5000, optimizer=MBGD(learningRate=0.01))

net.loadParamsFromFile("/home/ayush/scratch/Net/aknet/serialized.json")

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)

net.serialize("serialized.json")