from tensor import Tensor
import numpy as np

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        predicted = np.clip(predicted,a_min=-1e70,a_max = 1e70)
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class CrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        idx: int = np.argmax(actual)
        val: float = predicted[idx]
        val = np.max(np.array([val,np.finfo(float).eps]))
        return -np.log(val)
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return predicted - actual