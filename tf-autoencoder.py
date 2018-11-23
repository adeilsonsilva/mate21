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
import matplotlib.pyplot as plt
import tensorflow as tf


MAX_EPOCHS = 50

IMG_HEIGHT = 100
IMG_WIDTH = 100
NUM_CHANNELS = 100

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Load a database and train on it with tensorflow.")
    parser.add_argument("-p", "--path", help="Path to dataset.", required=True)
    parser.add_argument("-vp", "--validation_percentage", type=float, default=0.25, help="Percentage of training set used as validation.")
    parser.add_argument("-me", "--max_epochs", type=int, default=MAX_EPOCHS, help="Number of epochs to be run.")
    return parser.parse_args(argv)


def main(args):
    print("\t\t ** LOADING IMAGES **")
    training_images, validation_images = common_data.load_training_images(args.path, args.validation_percentage, as_vector = False, return_paths = True)
    print("\t\t ** {} {} **".format(training_images.shape, validation_images.shape))
    print("\t\t ** TRAINING MODEL **")
    train(training_images, validation_images, args.path, max_epochs = args.max_epochs)
    print("\t\t ** EXITING **")


"""
    Create tensorflow graph.
"""
graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, shape = (None, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    # y = tf.placeholder(tf.int64, shape = (None,))
    # y_one_hot = tf.one_hot(y, N_CLASS)
    learning_rate = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    print("================================================")
    print("X: {}".format(X.shape))

    # Input layer
    # out = tf.layers.batch_normalization(X, training=is_training)
    # out = tf.layers.dropout(out, 0.25)
    print("Input Layer: {}".format(out.shape))

    # CNN layers
    out = tf.layers.conv2d(out, 4, (8, 8), (1, 1), padding='same')
    print("Conv. Layer 1: {}".format(out.shape))
    out = tf.layers.conv2d(out, 8, (5, 5), (5, 5), padding='valid')
    print("Conv. Layer 2: {}".format(out.shape))
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), padding='valid')
    print("Max Pooling Layer 2: {}".format(out.shape))

    # Output layer
    out = tf.layers.dense(out, N_CLASS, activation=tf.nn.softmax)
    print("Output Layer: {}".format(out.shape))

    # Define loss/cost function and optimizer
    loss = tf.reduce_mean(tf.square(tf.subtract(output, X)))
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.485762).minimize(loss)

    print("================================================")

def run_validation(session, images):
    # Number of images
    N = images.shape[0]

    my_X, my_y = common_data.load_batch(images, augmentation=False)

    eval_loss = tf_session.run(tf_loss, feed_dict = {X: my_X, is_training: False})

    images = np.concatenate((my_X[0], output), axis=1)
    cv2.imshow("Result", images)
    cv2.waitKey(5000)

    return eval_loss/N

def train(training_images, validation_images, path, l_rate = common_data.L_RATE, max_epochs = common_data.MAX_EPOCHS):

    out_plot = open(PLOT_DATA_OUTPUT,"w")

    training_set_size = len(training_images)

    # Init tensorflow
    with tf.Session(graph = graph) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(max_epochs):
            print("{} -> Epoch {}".format(datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"), epoch+1))

            # Mini-batch gradient descent
            np.random.shuffle(training_images)
            train_acc = 0
            train_loss = 0
            for idx in range(0, training_set_size, common_data.BATCH_SIZE):
                # Load images only when used
                my_X, my_y = common_data.load_batch(training_images[idx:idx+common_data.BATCH_SIZE], augmentation=False)
                _, my_loss = session.run([optimizer, loss], feed_dict = {X: my_X, Y: my_y, learning_rate: l_rate, is_training: True})
                train_loss += my_loss

            print("\t *TRAINING* SET => LOSS: {:.4f}".format(train_loss/training_set_size))
            validation_loss = run_validation(session, validation_images)
            print("\t *VALIDATE* SET => LOSS: {:.4f}".format(validation_loss))

if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
