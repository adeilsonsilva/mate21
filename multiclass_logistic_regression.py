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

# https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html
N_CLASS = 10

# Max number of epochs to run
MAX_EPOCHS = 50000

# Image dimensions
IMG_WIDTH = 71
IMG_HEIGHT = 77
# There are 5000 training images
DATA_SIZE = 5000

# Learning rate
L_RATE = 1e-3

# Stop training if loss variates less than this
MAX_DLOSS = 1e-5

# Batch gradient descent parameter
# https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3
BATCH_SIZE = 32

# Output file name
FILENAME = "multiclass_logistic_regression.json"
OUTPUT_FILENAME = "multiclass_logistic_regression.output"

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
    return one_hot

def linear_regression(x, w, b):
    y = np.dot(x, w) + b
    return y

def softmax(z):
    ez = np.exp(z)
    return ez/np.sum(ez)

# Cost function
def cross_entropy(yhat, one_hot):
    # return -(np.sum(one_hot * np.log(yhat+1e-6)))
    return -(np.sum(one_hot * np.log(yhat)))

# It returns a vector with each position being the probability of being from thath class
def net(x, w, b):
    y_linear = linear_regression(x, w, b)
    yhat = softmax(y_linear)
    return yhat

def gradient_descent_step(b0, w0, images, learning_rate=L_RATE):
    # compute gradients
    w_grad = np.ones((N_CLASS, IMG_WIDTH*IMG_HEIGHT))
    b_grad = np.zeros(N_CLASS)
    N = len(images)
    cumulative_loss = 0
    scores = []
    for i in range(N):
        # Observation
        x = images[i][0]
        # Label
        y = images[i][1]
        label_one_hot = create_one_hot(y, N_CLASS)
        # Compute linear regression with initial weight/bias
        # We transpose weight matrix because of its dimensions
        y_probs = net(x, w0.T, b0)
        y_probs_loss = cross_entropy(y_probs, label_one_hot)
        cumulative_loss += np.sum(y_probs_loss)
        # https://madalinabuzau.github.io/2016/11/29/gradient-descent-on-a-softmax-cross-entropy-cost-function.html
        dscores = y_probs
        dscores[y] -= 1
        dscores /= N
        scores.append(dscores)
        # b_grad is the derivative of the loss in the direction of the bias
        b_grad += dscores

    # Cross-entropy is not element-wise
    X = np.array([images[i][0] for i in range(N)])
    # w_grad is the derivative of the loss in the direction of the weights
    w_grad = np.dot(X.T, np.array(scores)).T

    # update parameters
    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)

    return b1, w1, (cumulative_loss/N)

# It returns a class for that image
def predict(image, weights, bias):
    y_probs = net(image, weights, bias)
    return np.argmax(y_probs)

def get_accuracy(bias, weights, images):
    ACC = 0
    N = len(images)
    for i in range(N):
        x = images[i][0]
        y = images[i][1]
        # Don't forget to transpose weights vector
        y_predicted = predict(x, weights.T, bias)
        # print("\t\t {} : {}".format(y, y_predicted))
        if y == y_predicted:
            ACC += 1
    return ACC / N

def train(training_images, validation_images, learning_rate=L_RATE, max_epochs=MAX_EPOCHS):
    # Generate random weights and bias
    W = np.zeros((N_CLASS, IMG_WIDTH*IMG_HEIGHT))
    B = np.zeros(N_CLASS)

    # Used to control if we are getting closer to desired loss
    prev_loss = 0
    # We wanna save the model with best accuracy
    max_acc = 0
    max_acc_bias = 0
    max_acc_weights = []
    max_acc_loss = 0
    max_acc_epoch = 0

    for epoch in range(max_epochs):
        print("* Epoch {}".format(epoch+1))

        # Mini-batch gradient descent
        np.random.shuffle(training_images)
        B, W, loss = gradient_descent_step(B, W, training_images[0:BATCH_SIZE])
        accuracy = get_accuracy(B, W, validation_images)

        print("+++ ACC: {}".format(accuracy))
        if accuracy >= max_acc:
            max_acc = accuracy
            max_acc_bias = B
            max_acc_weights = W
            max_acc_loss = loss
            max_acc_epoch = epoch

        print("--- LOSS: {}".format(loss))
        d_loss = abs(loss - prev_loss)
        if d_loss <= MAX_DLOSS:
            print("\t\t ** CONVERGED!")
            break
        prev_loss = loss

    if epoch >= max_epochs-1:
        print("\t\t %% MAX EPOCHS REACHED!! %%")

    # model = {
    #     'bias': B.tolist(), 'weights': W.tolist()
    # }

    model = {
        'last': {'acc': accuracy, 'bias': B.tolist(), 'weights': W.tolist(), 'loss': loss, 'epoch': epoch},
        'best': {'acc': max_acc, 'bias': max_acc_bias.tolist(), 'weights': max_acc_weights.tolist(), 'loss': max_acc_loss, 'epoch': max_acc_epoch}
    }

    return model

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
