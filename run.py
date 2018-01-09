# run NN code

from data import load_mnist_data
import nnet
import numpy as np

from matplotlib import pyplot as plt

def show(x):
    """ visualize a single training example """
    im = plt.imshow(np.reshape(1 - x, (28, 28)))
    im.set_cmap('gray')

print("loading MNIST dataset")
(train_data, valid_data) = load_mnist_data()

# reduce data sets for faster speed:
train_data = train_data
valid_data = valid_data

# to see a training example, uncomment:
#x, y = train_data[123]
#show(x)
#plt.title("label = %d" % y)

# some initial params, not necessarily good ones
net = nnet.Network([784, 80, 10])

print("training")
net.train(train_data, valid_data, epochs=10, mini_batch_size=8, alpha=0.5)

ncorrect = net.evaluate(valid_data)
print("Validation accuracy: %.3f%%" % (100 * ncorrect / len(valid_data)))
