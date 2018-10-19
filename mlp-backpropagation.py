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
FILENAME = "mlp.json"
OUTPUT_FILENAME = "mlp.output"

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Load a database and train on it.")
    parser.add_argument("-p", "--path", help="Path to dataset.", required=True)
    parser.add_argument("-vp", "--validation_percentage", type=float, default=0.25, help="Percentage of training set used as validation.")
    parser.add_argument("-me", "--max_epochs", type=int, default=MAX_EPOCHS, help="Number of epochs to be run.")
    return parser.parse_args(argv)

def save_model(model):
    json_result = json.dumps(model)
    f = open(FILENAME,"w")
    f.write(json_result)
    f.close()

def load_model():
    f = open(FILENAME,"r")
    json_result = json.loads(''.join(f.readlines()))
    f.close()
    return json_result

def load_training_images(path, validation_percentage):
    # Load all file paths

    # Get class labels for training
    # Filters out blank values, removes "\n", splits by "/" and get last position of strip.
    class_names = [ list(filter(None, i.rstrip().rsplit("/")))[-1] for i in os.popen("ls -1d {}/train/*/".format(path)).readlines() ]

    # 3D Arrays -> each position is an array with an image (which is an array) and a label
    # training_images[i][0] -> image
    # training_images[i][1] -> label
    training_images = []
    validation_images = []

    # Get images for each class and split into training and validation
    images_counter = 0
    for idx in range(0, len(class_names)):
        # Get all
        class_images = [ i.rstrip() for i in os.popen("ls {}train/{}/* | sort -V".format(path, class_names[idx])).readlines() ]
        class_images_len = len(class_images)
        images_counter += class_images_len

        # Split into training and validation based on idx
        validation_idx = int(np.floor(class_images_len * validation_percentage))

        # Load training images
        for image_path in class_images[:class_images_len - validation_idx - 1]:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
            image = image.reshape(IMG_WIDTH*IMG_HEIGHT) / 255.0
            training_images.append([ image, idx ])

        # Load validation images
        for image_path in class_images[class_images_len - validation_idx:]:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
            image = image.reshape(IMG_WIDTH*IMG_HEIGHT) / 255.0
            validation_images.append([ image, idx ])

    return training_images, validation_images

# A one_hot vector is all zeroed with a 1 on the class label for that observation
def create_one_hot(iClass, number_of_classes):
    one_hot = np.zeros(number_of_classes)
    one_hot[iClass] = 1
    # for i in range(number_of_classes):
    #     one_hot[i] = 1e-2
    # one_hot[iClass] = 1 - ((number_of_classes - 1) * 1e-2)
    return one_hot

def linear_regression(x, w, b):
    # print(x.shape)
    y = np.dot(x, w) + b
    return y

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
def sigmoid_derivative(output):
    return sigmoid(output) * (1 - sigmoid(output))

def loss_function(y_predicted, y):
    # N = len(y)
    # print(np.sum((np.array(y_predicted) - np.array(y))**2))
    loss = np.sum((np.array(y_predicted) - np.array(y)) ** 2) / N_CLASS
    return loss

def loss_function_derivative(output, expected):
    return (output - expected) * sigmoid_derivative(output)

def softmax(z):
    ez = np.exp(z)
    return ez/np.sum(ez)

class Layer(object):
    """
        This is a Neural Network layer.
    """

    def __init__(self, input_size, output_size, role='hidden'):
        self.role = role
        self.weights = np.full((input_size, output_size), 1/output_size)
        self.bias = np.full((output_size), 1/output_size)

    # X is layer's input
    # z is layer's linear regression
    # a is layer's output (after activation)
    def forward(self, X):
        # print(self.role + "-->")
        self.X = X
        if self.role == 'output':
            self.A = self.activation(X)
        else:
            self.Z = linear_regression(X, self.weights, self.bias)
            self.A = self.activation(self.Z)
        return self.A

    def backward(self, expected):
        if self.role == 'output':
            self.dA = loss_function_derivative(self.A, expected)
        else:
            # Modo correto mas não roda por conta das dimensões
            # self.dZ = sigmoid_derivative(self.Z)
            # self.dA = self.dZ * np.dot(expected, self.weights.T)
            # self.dW = self.dA * self.X
            # self.dB = np.sum(self.dA)

            # Assim funciona mas está incorreto
            self.dZ = sigmoid_derivative(self.Z)
            self.dA = np.sum(self.dZ) * np.dot(expected, self.weights.T)
            self.dW = self.dA * self.X
            self.dB = np.sum(self.dA)

        return self.dA

    def update(self, learning_rate=L_RATE):
        if self.role != 'output':
            # Modo correto mas não roda por conta das dimensões
            # self.weights -= (learning_rate * self.dW)
            # Assim funciona
            for i in range(N_CLASS):
                self.weights[:,i] -= (learning_rate * self.dW)
            self.bias -= (learning_rate * self.dB)
        return

    def activation(self, z):
        if self.role == 'hidden':
            return sigmoid(z)
        return z
        # elif self.role == 'output':
        #     # output layer will give out a one_hot as output
        #     return create_one_hot(np.argmax(z), N_CLASS)
            # print("z: {}".format(z))
            # return np.argmax(z)


def net_forward(model, image):
    a = image
    # a = model['input_layer'].forward(image)
    for layer in model['hidden_layers']:
        a = layer.forward(a)
    prediction = model["output_layer"].forward(a)

    return prediction

def net_backward(model, output, expected):
    dA = model["output_layer"].backward(expected)
    model["output_layer"].update()
    # reverse list
    for layer in model['hidden_layers'][::-1]:
        dA = layer.backward(dA)
        layer.update()
    # model["input_layer"].backward(dA)
    # model["input_layer"].update()

    return

def get_accuracy(model, images):
    ACC = 0
    N = len(images)
    for i in range(N):
        x = images[i][0]
        y = images[i][1]
        # Don't forget to transpose weights vector
        y_predicted = net_forward(model, x)
        # print("\t\t {} : {}".format(y, y_predicted))
        # y_predicted is a one_hot vector
        # if y_predicted == y:
        # if y_predicted[y] == 1:
        if np.argmax(y_predicted) == y:
            ACC += 1
    return ACC / N

def train(training_images, validation_images, learning_rate=L_RATE, max_epochs=MAX_EPOCHS, hidden_layers=1):

    model = {
        # "input_layer": Layer(N_FEATURES, N_CLASS, role='input'),
        "hidden_layers": [],
        "output_layer": Layer(N_CLASS, 1, role='output'),
    }

    for l in range(hidden_layers):
        model["hidden_layers"].append(Layer(N_FEATURES, N_CLASS))

    # Used to control if we are getting closer to desired loss
    prev_loss = 0
    # We wanna save the model with best accuracy
    max_acc = 0
    max_acc_bias = 0
    max_acc_weights = []
    max_acc_loss = 0
    max_acc_epoch = 0

    # Mini-batch gradient descent
    np.random.shuffle(training_images)
    batch = training_images[0:BATCH_SIZE]
    out_plot = open('mlp.csv',"w")

    for epoch in range(max_epochs):
        print("* Epoch {}".format(epoch+1))

        # online learning
        loss = 0
        for observation in batch:
            image = observation[0]
            label = observation[1]
            # print("FORWARD")
            output = net_forward(model, image)
            # expected = label
            expected = create_one_hot(label, N_CLASS)
            # print("{} :: {}".format(output, expected))
            loss += loss_function(output, expected)
            # print("BACKWARD")
            net_backward(model, output, expected)
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
        accuracy = get_accuracy(model, validation_images)
        out_plot.write("{}, {}, {}\n".format(epoch, loss, accuracy))
        print("+++ ACC: {}".format(accuracy))
        print("--- LOSS: {}".format(loss))

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

def test(path, bias, weights):

    paths = [ i.rstrip() for i in os.popen("ls {}test/* | sort -V".format(path)).readlines() ]
    names = [ list(filter(None, i.rstrip().rsplit("/")))[-1] for i in os.popen("ls {}test/*".format(path)).readlines() ]

    with open(OUTPUT_FILENAME, "w") as f:
        nimg = 0
        for image_path in paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
            image = image.reshape(IMG_WIDTH*IMG_HEIGHT) / 255.0

            # Don't forget to transpose weights vector
            y_predicted = predict(image, weights.T, bias)

            f.write("{} {}\n".format(names[nimg], y_predicted))
            nimg += 1

    f.close()

def main(args):
    print("\t\t ** LOADING IMAGES **")
    training_images, validation_images = load_training_images(args.path, args.validation_percentage)
    print("\t\t ** TRAINING MODEL **")
    model = train(training_images, validation_images, max_epochs = args.max_epochs)
    print("\t\t ** SAVING MODEL **")
    save_model(model)
    print("\t\t ** TESTING **")
    test(args.path, np.array(model['best']['bias']), np.array(model['best']['weights']))

if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
