import json
from tensor import Tensor
from layer import Layer, LinearLayer, ActivationLayer
from loss import CrossEntropy
from typing import Sequence, Iterator, Tuple
import numpy as np



class NeuralNetwork:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    def backwards(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backwards(grad)
        return grad
    def getParamsAndGrads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for key in layer.params:
                yield layer.params[key], layer.grads[key+"Grad"]
    def setGradsZero(self) -> None:
        for layer in self.layers:
            for key in layer.grads:
                layer.grads[key].fill(0)
    def serialize(self, file: str) -> None:
        serL = []
        for index, layer in enumerate(self.layers):
            ser = {}
            ser["index"] = index
            if "weights" in layer.params and "bias" in layer.params:
                ser["weights"] = [list(x) for x in layer.params["weights"]]
                ser["bias"] = list(layer.params["bias"])
            serL.append(ser)
        serL = json.dumps(serL, indent=4)
        with open(file, "w") as f:
            f.write(serL)
    def loadParamsFromFile(self, file: str) -> None:
        with open(file, "r") as f:
            data = f.read()
        data = json.loads(data)
        for diction in data:
            if len(diction) == 1:
                continue           
            else:
                idx = diction["index"]
                self.layers[idx].params["weights"] = np.array(diction["weights"])
                self.layers[idx].params["bias"] = np.array(diction["bias"])
                
        
