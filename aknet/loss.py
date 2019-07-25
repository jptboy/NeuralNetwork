from tensor import Tensor
import numpy as np

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class CrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        idx: int = np.argmax(actual)
        val: float = predicted[idx]
        return -np.log(val)
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return predicted - actual