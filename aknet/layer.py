'''
Layer for neural network
'''
from typing import Dict, Callable
import numpy as np

from tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {} 
        self.grads: Dict[str, Tensor] = {} 
    def forward(self, input: Tensor):
        raise NotImplementedError
    def backwards(self, inputGrad: Tensor):
        raise NotImplementedError   
class LinearLayer(Layer):
    def __init__(self, inputSize, outputSize) -> None:
        super().__init__()
        self.params["weights"] = np.random.rand(outputSize,inputSize) 
        self.params["bias"] = np.random.rand(outputSize)
        self.grads["biasGrad"] = np.zeros((outputSize)) 
        self.grads["weightsGrad"] = np.zeros((outputSize,inputSize)) 
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return (self.params["weights"] @ input) + self.params["bias"]
    def backwards(self, inputGrad: Tensor) -> Tensor:
        self.grads["biasGrad"] += inputGrad
        deltaMatr: Tensor = np.column_stack(np.tile(inputGrad,(self.input.size,1))) 
        activGrad: Tensor = np.tile(self.input,(inputGrad.size,1)) 
        self.grads["weightsGrad"] += deltaMatr * activGrad
        return self.params["weights"].T @ inputGrad

class ActivationLayer(Layer):
    def __init__(self, f: Callable[[Tensor], Tensor], fPrime: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.f = f
        self.fPrime = f
    def forward(self,input: Tensor) -> Tensor:
        self.input = input
        return self.f(input)
    def backwards(self,grad: Tensor) -> Tensor:
        fPrimeGrad = self.fPrime(self.input) 
        if len(np.shape(fPrimeGrad)) > 1:
            return fPrimeGrad @ grad
        return  fPrimeGrad * grad

class Sigmoid(ActivationLayer):
    def __init__(self) -> None:
        super().__init__(sigmoid,sigmoidPrime)

class Softmax(ActivationLayer):
    def __init__(self) -> None:
        super().__init__(softmax,softmaxPrime)

class Relu(ActivationLayer):
    def __init__(self) -> None:
        super().__init__(relu,reluPrime)
class Tanh(ActivationLayer):
    def __init__(self) -> None:
        super().__init__(tanh,tanhPrime)
def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanhPrime(x: Tensor) -> Tensor:
    return 1. - x * x

def softmax(v: Tensor) -> Tensor:
    exps = np.exp(v)
    tot = np.sum(exps)
    return exps / tot

def softmaxPrime(v: Tensor) -> Tensor:
    s = softmax(v)
    s = s.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def sigmoid(v: Tensor) -> Tensor:
    return 1/(1 + np.exp(-v))

def sigmoidPrime(v: Tensor) -> Tensor:
    x = sigmoid(v)
    return x * (1-x)

def relu(x: Tensor) -> Tensor:
    return x * (x > 0)

def reluPrime(x: Tensor) -> Tensor:
    return 1. * (x > 0)