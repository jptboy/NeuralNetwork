from network import NeuralNetwork
from layer import LinearLayer, ActivationLayer


class Optimizer:
    def step(self, net: NeuralNetwork, percent:int) -> None:
        raise NotImplementedError

class MBGD(Optimizer):
    def __init__(self, learningRate = 0.00025):
        self.learningRate = learningRate
        self.inc = self.learningRate/500000
    def step(self, net: NeuralNetwork, percent: int = 2.5) -> None:
        # self.learningRate += self.inc
        for param, grad in net.getParamsAndGrads():
            param -= self.learningRate * grad
        net.setGradsZero()
