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

import numpy as np

# ----------------------------------------------------- #
# Weighted sum.                                         #
# ----------------------------------------------------- #
def linear_regression(x, w, b):
    y = np.dot(x, w) + b
    return y

# ----------------------------------------------------- #
# Activation/transfer functions and their derivatives.  #
# ----------------------------------------------------- #
def softmax(z):
    ez = np.exp(z)
    return ez/np.sum(ez)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
def sigmoid_derivative(output):
    sig = sigmoid(output)
    return sig * (1 - sig)

# ----------------------------------------------------- #
# Loss/cost functions.                                  #
# ----------------------------------------------------- #
def cross_entropy(yhat, one_hot):
    return -(np.sum(one_hot * np.log(yhat)))

def MSE(y_predicted, y):
    N = len(y_predicted)
    return (1/N)*(((y_predicted - y)**2))

def MSE_sigmoid_derivative(activation, Z, expected):
    return (activation - expected) * sigmoid_derivative(Z)

# ----------------------------------------------------- #
# Helper functions                                      #
# ----------------------------------------------------- #

def create_one_hot(iClass, number_of_classes):
    """
        A one_hot vector is all zeroed with a 1 on the class label for that observation
    """
    one_hot = np.zeros(number_of_classes)
    one_hot[iClass] = 1
    return one_hot
