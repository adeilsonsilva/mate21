import argparse
import cv2
import numpy as np
import os

MAX_EPOCHS = 5000
IMG_WIDTH = 71
IMG_HEIGHT = 77
L_RATE=1e-3

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Load a database and train on it.")
    parser.add_argument("-p", "--path", help="Path to dataset.", required=True)
    parser.add_argument("-vp", "--validation_percentage", type=float, default=0.25, help="Percentage of training set used as validation.")
    parser.add_argument("-me", "--max_epochs", type=int, default=MAX_EPOCHS, help="Number of epochs to be run.")
    return parser.parse_args(argv)

def load_images(path, validation_percentage):
    # Load all file paths

    # Get class labels for training
    # Filters out blank values, removes "\n", splits by "/" and get last position of strip.
    class_names = [ filter(None, i.rstrip().rsplit("/"))[-1] for i in os.popen("ls -1d {}/train/*/".format(path)).readlines() ]

    # Get images for each class and split into training and validation
    dataset_paths = {}
    # training_images_paths = {}
    training_images = []
    validation_images = []
    training_labels = []
    validation_labels = []
    # validation_images_paths = {}
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
            # height, width = image.shape
            image = image.reshape(1, -1)
            training_images.append(image)
            training_labels.append(idx)
        # Load validation images
        for image_path in class_images[class_images_len - validation_idx:]:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = image.reshape(1, -1)
            validation_images.append(image)
            validation_labels.append(idx)
    return training_images, validation_images, training_labels, validation_labels

def gradient_descent_step(b0, w0, images, labels, learning_rate=L_RATE):
    # compute gradients
    b_grad = 0
    w_grad = 0
    N = len(images)
    for i in range(N):
        x = images[i]
        y = labels[i]
        b_grad += (2.0/N)*(w0*x + b0 - y)
        w_grad += (2.0/N)*x*(w0*x + b0 - y)

    # update parameters
    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)

    return b1, w1

def epoch():
    return

def train(training_images, validation_images, training_labels, validation_labels, learning_rate=L_RATE, max_epochs=MAX_EPOCHS):
    # Generate random weights
    weights = np.random.rand(1, IMG_WIDTH*IMG_HEIGHT)
    bias = 1
    for epoch in range(max_epochs):
        print("* Epoch {}".format(epoch+1))
        for image in training_images:
            new_bias, new_weights = gradient_descent_step(bias, weights, training_images, training_labels)
            bias += new_bias * learning_rate
            weights += new_weights * learning_rate

    return {'bias': bias, 'weights': weights}

def main(args):
    training_images, validation_images, training_labels, validation_labels = load_images(args.path, args.validation_percentage)
    model = train(training_images, validation_images, training_labels, validation_labels)

if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
