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

import cv2
import json
import numpy as np
import os

# https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html
N_CLASS = 10

# Max number of epochs to run
MAX_EPOCHS = 50

# Image dimensions
IMG_WIDTH = 71
IMG_HEIGHT = 77
NUM_CHANNELS = 1

# There are 5000 training images
DATA_SIZE = 5000

# Learning rate
L_RATE = 1e-1

# Stop training if loss variates less than this
MAX_DLOSS = 1e-7
MIN_STD = 1e-3

# Batch gradient descent parameter
# https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3
BATCH_SIZE = 32

def save_model(model, model_path):
    json_result = json.dumps(model)
    f = open(model_path,"w")
    f.write(json_result)
    f.close()

def load_model(model_path):
    f = open(model_path,"r")
    json_result = json.loads(''.join(f.readlines()))
    f.close()
    return json_result

def load_training_images(path, validation_percentage, as_vector = True):
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
            if as_vector:
                image = image.reshape(IMG_WIDTH*IMG_HEIGHT*NUM_CHANNELS) / 255.0
            else:
                image = image.reshape(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS) / 255.0
            training_images.append([ image, idx ])

        # Load validation images
        for image_path in class_images[class_images_len - validation_idx:]:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
            if as_vector:
                image = image.reshape(IMG_WIDTH*IMG_HEIGHT*NUM_CHANNELS) / 255.0
            else:
                image = image.reshape(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS) / 255.0
            validation_images.append([ image, idx ])

    return np.array(training_images), np.array(validation_images)
