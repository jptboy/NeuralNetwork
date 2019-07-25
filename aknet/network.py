from tensor import Tensor
from layer import *
from loss import CrossEntropy
from typing import Sequence



class NeuralNetwork:
    def __init__(layers: Sequence[Layer]) -> None:
        self.layers = layers
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    def backwards(self, grad: Tensor) -> Tensor:
