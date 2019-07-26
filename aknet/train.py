from tensor import Tensor
from matplotlib import pyplot as plt
from network import NeuralNetwork
from loss import CrossEntropy, Loss, MSE
from optimizer import Optimizer, MBGD
from dataIterator import DataIterator, BatchIterator


def train(net: NeuralNetwork,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = CrossEntropy(),
          optimizer: Optimizer = MBGD(),
          showGraph: bool = False) -> None:
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            for X, Y in zip(batch.inputs, batch.targets):
                predicted = net.forward(X)
                epoch_loss += loss.loss(predicted, Y)
                grad = loss.grad(predicted, Y)
                net.backwards(grad)
                optimizer.step(net)
        print(epoch, epoch_loss)
        losses.append(epoch_loss)
    if showGraph:
        plt.plot(losses)
        plt.show()
