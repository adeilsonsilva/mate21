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
import random

# https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html
N_CLASS = 10

# Max number of epochs to run
MAX_EPOCHS = 1

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
BATCH_SIZE = 64

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

# Generate a random number and apply the change
def augmentate(image):

    # Rotates by an angle of [1, 5]
    if (random.random() >= 0.5):
        angle = random.randint(1, 15)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Flips horizontally
    # if (random.random() >= 0.5):
    #     cv2.flip(image, 1, image)

    # Gaussian noise
    # if (random.random() >= 0.5):
    #     noise =  image.copy()
    #     cv2.randn(noise, (0), (150))
    #     image += noise

    # Uniform noise
    if (random.random() >= 0.5):
        noise =  image.copy()
        cv2.randu(noise, (0), (1))
        image += noise

    # print(image.shape)
    # cv2.imshow('teste', image)
    # cv2.waitKey(5000)
    return image

# This function receives a list of image paths, loads them with OpenCV
# and return an array with images and their labels
def load_batch(batch, as_vector=False, augmentation=False):
    images = []
    labels = []
    for tuple in batch:
        image_path = tuple[0]
        label = tuple[1]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if augmentation:
            # Stochastic augmentation
            image = augmentate(image)
        if as_vector:
            # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
            image = image.reshape(IMG_WIDTH*IMG_HEIGHT*NUM_CHANNELS) / 255.0
        else:
            image = image.reshape(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS) / 255.0
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

def load_training_images(path, validation_percentage, as_vector = True, return_paths = False):
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
        # Shuffle list to get different images each time
        random.shuffle(class_images)
        class_images_len = len(class_images)
        images_counter += class_images_len

        # Split into training and validation based on idx
        validation_idx = int(np.floor(class_images_len * validation_percentage))

        # Get splits paths
        training_paths = class_images[:class_images_len - validation_idx]
        validation_paths = class_images[class_images_len - validation_idx:]

        # Load training images
        for image_path in training_paths:
            if return_paths:
                image = image_path
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if as_vector:
                    # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
                    image = image.reshape(IMG_WIDTH*IMG_HEIGHT*NUM_CHANNELS) / 255.0
                else:
                    image = image.reshape(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS) / 255.0
            training_images.append([ image, idx ])

        # Load validation images
        for image_path in validation_paths:
            if return_paths:
                image = image_path
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if as_vector:
                    # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
                    image = image.reshape(IMG_WIDTH*IMG_HEIGHT*NUM_CHANNELS) / 255.0
                else:
                    image = image.reshape(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS) / 255.0
            validation_images.append([ image, idx ])

    return np.array(training_images), np.array(validation_images)
