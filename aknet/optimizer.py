from network import NeuralNetwork
from layer import LinearLayer, ActivationLayer


class Optimizer:
    def step(self, net: NeuralNetwork) -> None:
        raise NotImplementedError

class MBGD(Optimizer):
    def __init__(self, learningRate = 0.01):
        self.learningRate = learningRate
    def step(self, net: NeuralNetwork) -> None:
        for param, grad in net.getParamsAndGrads():
            param -= self.learningRate * grad
        net.setGradsZero()
