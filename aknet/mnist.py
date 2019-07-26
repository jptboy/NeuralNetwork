# from tensorflow.keras.datasets import mnist
# dataMnist = mnist.load_data() 
# train, test = dataMnist
# inputs, targets = train
# import numpy as np
# testx, testy = test

# import numpy as np
# inputs1 = []
# trainx1 = []
# targets1 = []
# for i in range(len(inputs)):
#     inputs1.append(inputs[i].flatten())
# for i in range(len(testx)):
#     trainx1.append(testx[i].flatten())
# for i in range(len(targets)):
#     foo = np.zeros(10)
#     foo[targets[i]] = 1
#     targets1.append(foo)

# inputs = np.array(inputs1)
# testx = np.array(trainx1)
# targets = np.array(targets1)



from train import train
from network import NeuralNetwork
from layer import LinearLayer, Sigmoid, Relu, Softmax, Tanh, LeakyRelu
from loss import MSE, CrossEntropy
from optimizer import MBGD
import numpy as np

from sklearn.datasets import load_digits
digits = load_digits()
inputs = digits.data
for x in inputs:
    x /= 255
targets = []
for num in digits.target:
    baz = np.zeros(10)
    baz[num] = 1
    targets.append(baz)
targets = np.array(targets)
from sklearn.model_selection import train_test_split
inputs,xtest,targets, ytest = train_test_split(inputs, targets, test_size = 0.2, random_state = 42)

np.seterr(all='raise')
net = NeuralNetwork([
    LinearLayer(inputSize=64, outputSize=128),
    LeakyRelu(),
    LinearLayer(inputSize=128, outputSize=10),
    LeakyRelu(),
    Softmax()
])

train(net, inputs, targets, loss= CrossEntropy(), num_epochs=20000, optimizer=MBGD(learningRate=0.001), showGraph=True)

net.serialize("serializedMNIST.json")
# net.loadParamsFromFile("/home/ayush/scratch/Net/aknet/serializedMNIST.json")
total = len(xtest)
correct = 0
for x, y in zip(xtest, ytest):
    predicted = net.forward(x)
    if np.argmax(predicted) == np.argmax(y):
        correct +=1
    # plt.imshow(x.reshape((28,28)))
    # plt.show()
    print(np.argmax(predicted), np.argmax(y))
print(correct/total)


