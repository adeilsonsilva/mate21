#!/usr/bin/env python

##################################################
## Universidade Federal da Bahia
## 2018.2 - MATE21
##################################################
## GPLv3
##################################################
## Author: Adeilson Silva
## Mmaintainer: github.com/adeilsonsilva
## Email: adeilson@protonmail.com
##################################################

import argparse
import cv2
import json
import numpy as np
import os

# Image dimensions
IMG_WIDTH = 71
IMG_HEIGHT = 77
# There are 5000 training images
DATA_SIZE = 5000

# https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html
N_CLASS = 10
N_FEATURES = IMG_WIDTH*IMG_HEIGHT

# Max number of epochs to run
MAX_EPOCHS = 50000

# Learning rate
L_RATE = 1e-3

# Stop training if loss variates less than this
MAX_DLOSS = 1e-5

# Batch gradient descent parameter
# https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3
BATCH_SIZE = 32

# Output file name
MODEL_FILENAME = "mlp.json"
OUTPUT_FILENAME = "mlp.output"
PLOT_FILE = "mlp.csv"

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Load a database and train on it.")
    parser.add_argument("-p", "--path", help="Path to dataset.", required=True)
    parser.add_argument("-vp", "--validation_percentage", type=float, default=0.25, help="Percentage of training set used as validation.")
    parser.add_argument("-me", "--max_epochs", type=int, default=MAX_EPOCHS, help="Number of epochs to be run.")
    return parser.parse_args(argv)

class Network (object):
    """
        This is a Neural Network.
    """

    def __init__(self):
        self.layers = []
        self.gradients = []

    def add_layer(self, input_size, output_size, role='hidden'):
        layer = Layer(input_size, output_size, role)
        self.layers.append(layer)

    def forward(self, image):
        # Image is inputted into first layer
        activation = self.layers[0].forward(image)
        for layer in self.layers[1:]:
            activation = layer.forward(activation)

        return activation

    def backward(self, expected):
        dA = self.layers[-1].backward(expected)
        # self.gradients.append(dA)
        # reverse list, excluding last position
        for layer in self.layers[::-1][1:]:
            dA = layer.backward(dA)
            # self.gradients.append([layer.dW, layer.dB])

        # return self.gradients

    def update(self):
        # print("G: {}".format(len(self.gradients)))
        # Update all layers but output's
        for layer in self.layers[:-1]:
            layer.update()
        # self.gradients = []

    def get_accuracy(self, images):
        ACC = 0
        N = len(images)
        for i in range(N):
            x = images[i][0]
            y = images[i][1]
            # Don't forget to transpose weights vector
            y_predicted = self.forward(x)
            # y = create_one_hot(y, N_CLASS)
            # print("\t\t {} : {}".format(y, y_predicted))
            # y_predicted is a one_hot vector
            # if (y_predicted == y).all():
            # if y_predicted[y] == 1:
            if np.argmax(y_predicted) == y:
                ACC += 1
        return ACC / N

    def run_batch(self, batch):
        # Used to control if we are getting closer to desired loss
        prev_loss = 0
        # We wanna save the model with best accuracy
        max_acc = 0
        max_acc_bias = 0
        max_acc_weights = []
        max_acc_loss = 0
        max_acc_epoch = 0
        loss = 0
        self.restart_gradients()
        for observation in batch:
            image = observation[0]
            label = observation[1]
            # print("FORWARD")
            # output = net_forward(model, image)
            output = self.forward(image)
            expected = label
            # expected = create_one_hot(label, N_CLASS)
            # print("{} :: {}".format(output, expected))
            loss += loss_function(output, expected)
            # loss += cross_entropy(output, expected)
            # print("BACKWARD")
            # net_backward(model, output, expected)
            self.backward(expected)
            # if accuracy >= max_acc:
            #     max_acc = accuracy
            #     max_acc_bias = B
            #     max_acc_weights = W
            #     max_acc_loss = loss
            #     max_acc_epoch = epoch

            # d_loss = abs(loss - prev_loss)
            # if d_loss <= MAX_DLOSS:
            #     print("\t\t ** CONVERGED!")
            #     break
        print("--- LOSS: {}".format(loss))

    def restart_gradients(self):
        for layer in self.layers[:-1]:
            layer.dW = np.zeros(layer.weights.T.shape)
            layer.dB = np.zeros(layer.bias.shape)

    def get_info(self):
        print("This network has {} layers.".format(len(self.layers)))
        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            print("Layer {}: {}, {}x{}.".format(idx, layer.role, layer.neurons, layer.output))


class Layer(object):
    """
        This is a Neural Network layer.
    """

    def __init__(self, input_size, output_size, role):
        self.neurons = input_size
        self.output = output_size
        self.role = role
        if self.role != 'output':
            self.weights = np.full((input_size, output_size), 1/output_size)
            self.bias = np.full((output_size), 1/output_size)
            # self.weights = np.random.random((input_size, output_size))
            # self.bias = np.random.random((output_size))

    # X is layer's input
    # z is layer's linear regression
    # a is layer's output (after activation)
    def forward(self, X):
        # print(self.role + "-->")
        self.X = X
        if self.role == 'output':
            self.A = self.activation(X)
        else:
            # print("{} --> Z: {}".format(self.role, self.weights.shape))
            self.Z = linear_regression(X, self.weights, self.bias)
            # print("{} --> Z: {}".format(self.role, self.Z))
            self.A = self.activation(self.Z)
            # print("{} --> A: {}".format(self.role, self.A))
        return self.A

    def backward(self, expected):
        if self.role == 'output':
            self.dA = loss_function_derivative(self.A, self.X, expected)
            # print("=== {}".format(self.dA))
        # elif self.role == 'hidden':
        #     self.dZ = sigmoid_derivative(self.Z)
        #     # self.dA = self.dZ * np.dot(expected, self.weights.T)
        #     self.dA = self.X
        #     self.dW += np.dot(self.A, self.dZ)
        #     self.dB += self.dA
        else:
            self.dZ = sigmoid_derivative(self.Z)
            self.dA = self.dZ * np.dot(expected, self.weights.T)
            self.dW += np.dot(self.dA, self.X)
            self.dB += self.dA

        return self.dA

    def update(self, learning_rate=L_RATE):
        if self.role != 'output':
            # Modo correto mas não roda por conta das dimensões
            # print("role: {}".format(self.role))
            # print("before: {}".format(self.weights))
            self.weights -= (learning_rate * self.dW.T)
            # print("dW: {}".format(self.dW.T))
            # print("after: {}".format(self.weights))
            # print("b4: {}".format(self.bias))
            self.bias -= (learning_rate * self.dB)
            # print("after: {}".format(self.bias))
        return

    def activation(self, z):
        if self.role == 'hidden':
            return sigmoid(z)
        # elif self.role == 'input':
        #     return z / np.max(z)
        elif self.role == 'output':
        #     # output layer will give out a one_hot as output
        #     # print("z: {}".format(z))
            # return create_one_hot(np.argmax(z), N_CLASS)
            return np.argmax(z)
            # return softmax(z)
        return z

class NeuralNet:

    GRAD_CHECK_THRESH = 10e-8
    GRAD_CHECK_ITER = 9999999
    EPSILSON = 10e-4

    def __init__(self, _maxIter=250, _nHidden=15):
        # Set hyperparameters
        self.maxIter = _maxIter
        self.nHidden = _nHidden
        self.learningRate = np.linspace(0.5, 0.05, self.maxIter)
        self.momentum = 0.5
        self.enNesterov = False  # Nesterov accelerated gradient (https://arxiv.org/pdf/1212.0901v2.pdf)
        self.cost = []
        self.gradDiff = []
        self.minGrad = []
        self.dW1 = 0
        self.dW2 = 0

    def initNet(self, _nInput, _nOutput):
        self.nInput = _nInput
        self.nOutput = _nOutput
        self.W1 = np.random.rand(self.nInput, self.nHidden)/100
        self.W2 = np.random.rand(self.nHidden, self.nOutput)/100

    def train(self, X, y):
        # Initialise network structure
        self.initNet(X.shape[1], y.shape[1])

        # Train - full batch
        for m in range(self.maxIter):

            # Check gradient descent is operating as expected
            if m < self.GRAD_CHECK_ITER:
                self.checkGradients(X, y)

            # Train net
            yHat = self.feedforward(X)
            J = self.costFunction(y, yHat)
            self.cost.append(J)
            dJdW1, dJdW2 = self.backprop(X, y, yHat)
            self.updateWeights(dJdW1, dJdW2, self.learningRate[m])

    def feedforward(self, X):

        self.z2 = np.dot(X, self.W1)
        # print("X: {}".format(X.shape))
        # print("W1: {}".format(self.W1.shape))
        # print("z2: {}".format(self.z2.shape))
        self.a2 = self.sigmoid(self.z2)
        # print("a2: {}".format(self.a2.shape))
        self.z3 = np.dot(self.a2, self.W2)
        # print("z3: {}".format(self.z3.shape))
        yHat = self.sigmoid(self.z3)
        # print("yhat: {}".format(yHat.shape))
        return yHat

    def costFunction(self, y, yHat):
        J = 0.5 * np.sum((y - yHat)**2)
        return J

    def backprop(self, X, y, yHat):
        # NB: delta 3 depends on the cost function applied
        delta3 = np.multiply(-(y - yHat), self.sigmoidPrime(self.z3))
        print("self.sigmoidPrime(self.z3): {}".format(self.sigmoidPrime(self.z3)))
        print("d3: {}".format(delta3.shape))
        dJdW2 = np.dot(self.a2.T, delta3)
        print("self.a2.T: {}".format(self.a2.T.shape))
        print("dJdW2: {}".format(dJdW2.shape))
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        print("d2: {}".format(delta2.shape))
        dJdW1 = np.dot(X.T, delta2)
        print("dJdW1: {}".format(dJdW1.shape))
        return dJdW1, dJdW2

    def updateWeights(self, dJdW1, dJdW2, learnRate):
        if self.enNesterov:
            dW1_prev = self.dW1
            dW2_prev = self.dW2
            self.dW1 = learnRate*dJdW1 + self.momentum*self.dW1
            self.dW2 = learnRate*dJdW2 + self.momentum*self.dW2
            self.W1 = self.W1 - (1+self.momentum)*self.dW1 - self.momentum*dW1_prev
            self.W2 = self.W2 - (1+self.momentum)*self.dW2 - self.momentum*dW2_prev
        else:
            self.dW1 = learnRate*dJdW1 + self.momentum*self.dW1
            self.dW2 = learnRate*dJdW2 + self.momentum*self.dW2
            self.W1 = self.W1 - self.dW1
            self.W2 = self.W2 - self.dW2

    def getWeights(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def setWeights(self, weights):
        W1_start = 0
        W1_end = self.nInput * self.nHidden
        self.W1 = np.reshape(weights[W1_start:W1_end], (self.nInput, self.nHidden))
        W2_end = W1_end + self.nHidden * self.nOutput
        self.W2 = np.reshape(weights[W1_end:W2_end], (self.nHidden, self.nOutput))

    def checkGradients(self, X, y):
        # Numerical gradient
        numGrad = self.computeNumericalGradient(X, y)

        # Backprop gradient
        yHat = self.feedforward(X)
        dJdW1, dJdW2 = self.backprop(X, y, yHat)
        grad = np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
        self.minGrad.append(np.min(grad))

        # Compare
        diff = np.linalg.norm(grad - numGrad) / np.linalg.norm(grad + numGrad)
        if diff < self.GRAD_CHECK_THRESH:
            str = 'PASS'
        else:
            str = 'FAIL'
        print('[{0}] Gradient checking. diff = {1}'.format(str, diff))
        self.gradDiff.append(diff)

    def computeNumericalGradient(self, X, y):
        weights = self.getWeights()
        perturb = np.zeros(weights.shape)
        numGrad = np.zeros(weights.shape)

        for p in range(len(weights)):
            # Set pertubation for this weight only
            perturb[p] = self.EPSILSON
            # Positive perturbation
            self.setWeights(weights + perturb)
            yHat = self.feedforward(X)
            Jpos = self.costFunction(y, yHat)
            # Negative perturbation
            self.setWeights(weights - perturb)
            yHat = self.feedforward(X)
            Jneg = self.costFunction(y, yHat)
            # Compute Numerical Gradient
            numGrad[p] = (Jpos - Jneg) / (2 * self.EPSILSON)
            # Reset perturbation for next iteration
            perturb[p] = 0

        # Reset weights
        self.setWeights(weights)
        return numGrad

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        x = np.exp(-z)
        return (x / ((1 + x)**2))


def train(training_images, validation_images, learning_rate=L_RATE, max_epochs=MAX_EPOCHS, hidden_layers=1):

    network = Network()
    network.add_layer(N_FEATURES, 32, 'input')
    network.add_layer(32, N_CLASS, 'hidden')
    network.add_layer(N_CLASS, 1, 'output')
    network.get_info()

    # network = NeuralNet()
    # network.add_layer(N_FEATURES, 32, 'input')
    # network.add_layer(32, N_CLASS, 'hidden')
    # network.add_layer(N_CLASS, 10, 'output')
    # network.get_info()

    # Mini-batch gradient descent
    out_plot = open(PLOT_FILE,"w")

    for epoch in range(max_epochs):
        print("* Epoch {}".format(epoch+1))

        # online learning
        np.random.shuffle(training_images)
        batch = training_images[0:BATCH_SIZE]
        loss = 0
        # network.train(np.random.random((5, 500)), np.array([[1], [2], [3], [4], [5]]))
        network.run_batch(batch)
        accuracy = network.get_accuracy(validation_images)
        print("+++ ACC: {}".format(accuracy))
        # print("W_B4: {}".format(network.layers[1].weights))
        # print("dW_B4: {}".format(network.layers[1].dW))
        network.update()
        # print("W_AF: {}".format(network.layers[1].weights))
        out_plot.write("{}, {}, {}\n".format(epoch, loss, accuracy))

    if epoch >= max_epochs-1:
        print("\t\t %% MAX EPOCHS REACHED!! %%")

    # model = {
    #     'bias': B.tolist(), 'weights': W.tolist()
    # }
    out_plot.close()

    result = {
        'last': {'acc': accuracy, 'bias': B.tolist(), 'weights': W.tolist(), 'loss': loss, 'epoch': epoch},
        'best': {'acc': max_acc, 'bias': max_acc_bias.tolist(), 'weights': max_acc_weights.tolist(), 'loss': max_acc_loss, 'epoch': max_acc_epoch}
    }

    return result

def test(path, network):

    paths = [ i.rstrip() for i in os.popen("ls {}test/* | sort -V".format(path)).readlines() ]

    with open(OUTPUT_FILENAME, "w") as f:
        for image_path in paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
            image = image.reshape(IMG_WIDTH*IMG_HEIGHT) / 255.0

            # Don't forget to transpose weights vector
            y_predicted = net_forward(network, image)

            # Get only image name (e.g. '14734.png')
            img_name = image_path.rstrip().rsplit("/")[-1]
            f.write("{} {}\n".format(img_name, y_predicted))

    f.close()

def main(args):
    print("\t\t ** LOADING IMAGES **")
    training_images, validation_images = load_training_images(args.path, args.validation_percentage)
    print("\t\t ** TRAINING MODEL **")
    network = train(training_images, validation_images, max_epochs = args.max_epochs)
    print("\t\t ** SAVING MODEL **")
    save_network(network)
    print("\t\t ** TESTING **")
    test(args.path, network['best'])

if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
