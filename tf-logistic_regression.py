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

# My libraries
import common_math
import common_data

import argparse
import cv2
import datetime
import numpy as np
import os
import tensorflow as tf

PLOT_DATA_OUTPUT = "tf-logistic_regression.csv"
MODEL_FILENAME = "./tf-logistic_regression.model"
OUTPUT_FILENAME = "tf-logistic_regression.output"

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Load a database and train on it with tensorflow.")
    parser.add_argument("-p", "--path", help="Path to dataset.", required=True)
    parser.add_argument("-vp", "--validation_percentage", type=float, default=0.25, help="Percentage of training set used as validation.")
    parser.add_argument("-me", "--max_epochs", type=int, default=common_data.MAX_EPOCHS, help="Number of epochs to be run.")
    return parser.parse_args(argv)

def create_tf_graph():
    """
        Create tensorflow graph.
    """
    graph = tf.Graph()
    with graph.as_default():
    	X = tf.placeholder(tf.float32, shape = (None, common_data.IMG_HEIGHT*common_data.IMG_WIDTH*common_data.NUM_CHANNELS))
    	y = tf.placeholder(tf.int64, shape = (None,))
    	y_one_hot = tf.one_hot(y, common_data.N_CLASS)
    	learning_rate = tf.placeholder(tf.float32)

    	# fc = tf.layers.dense(X, 512, activation=tf.nn.relu)
    	out = tf.layers.dense(X, common_data.N_CLASS, activation=tf.nn.softmax)

    	# loss = tf.reduce_mean(tf.reduce_sum((y_one_hot-out)**2))
    	loss = tf.losses.softmax_cross_entropy(y_one_hot, out)

    	train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    	result = tf.argmax(out, 1)
    	correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))

    return graph, X, y, train_op, loss, correct, learning_rate, result

def get_accuracy(tf_session, images, tf_X, tf_y, tf_loss, tf_correct):
    # Number of images
    N = images.shape[0]

    my_X = np.array([ i for i in images[:,0] ])
    my_y = np.array([ i for i in images[:,1] ])

    eval_loss, eval_acc = tf_session.run([tf_loss, tf_correct], feed_dict = {tf_X: my_X, tf_y: my_y})

    return eval_acc/N, eval_loss/N

def train(training_images, validation_images, path, l_rate = common_data.L_RATE, max_epochs = common_data.MAX_EPOCHS):

    out_plot = open(PLOT_DATA_OUTPUT,"w")

    training_set_size = len(training_images)
    my_graph, tf_X, tf_y, tf_optimizer, tf_loss, tf_correct, tf_lr, tf_result = create_tf_graph()

    # Init tensorflow
    with tf.Session(graph = my_graph) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(max_epochs):
            print("* Epoch {} --> {}".format(epoch+1, datetime.datetime.now()))

            # Mini-batch gradient descent
            np.random.shuffle(training_images)
            train_acc = 0
            train_loss = 0
            for idx in range(0, training_set_size, common_data.BATCH_SIZE):
                batch = training_images[idx:idx+common_data.BATCH_SIZE]
                my_X = np.array([ i for i in batch[:,0] ])
                my_y = np.array([ i for i in batch[:,1] ])
                # my_X = batch[:,0]
                # my_y = batch[:,1]
                # print("&&&& {}".format(my_X))
                # print("&&&& {}".format(my_y))
                ret = session.run([tf_optimizer, tf_loss, tf_correct], feed_dict = {tf_X: my_X, tf_y: my_y, tf_lr: l_rate})
                train_loss += ret[1]
                train_acc += ret[2]

            print("\t *TRAINING* SET => ACC: {} ; LOSS: {}".format(train_acc/training_set_size, train_loss/training_set_size))
            validation_acc, validation_loss = get_accuracy(session, validation_images, tf_X, tf_y, tf_loss, tf_correct)
            print("\t *VALIDATION* SET => ACC: {} ; LOSS: {}".format(validation_acc, validation_loss))
            out_plot.write("{}, {}, {}, {}, {}\n".format(epoch, train_acc/training_set_size, train_loss/training_set_size, validation_acc, validation_loss))

        # Evaluate model on test set
        print("\t\t ** TESTING **")
        paths = [ i.rstrip() for i in os.popen("ls {}test/* | sort -V".format(path)).readlines() ]

        f = open(OUTPUT_FILENAME, "w")
        for image_path in paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Reshapes image to a 1D vector and normalizes all pixels to [0, 1]
            image = image.reshape(common_data.IMG_WIDTH*common_data.IMG_HEIGHT*common_data.NUM_CHANNELS) / 255.0

            y_predicted = session.run([tf_result], feed_dict = {tf_X: [image]})

            # Get only image name (e.g. '14734.png')
            img_name = image_path.rstrip().rsplit("/")[-1]
            f.write("{} {}\n".format(img_name, y_predicted[0][0]))

        f.close()
    out_plot.close()

    return

def main(args):
    print("\t\t ** LOADING IMAGES **")
    training_images, validation_images = common_data.load_training_images(args.path, args.validation_percentage)
    print("\t\t ** {} {} **".format(training_images.shape, validation_images.shape))
    print("\t\t ** TRAINING MODEL **")
    train(training_images, validation_images, args.path, max_epochs = args.max_epochs)
    # print("\t\t ** SAVING MODEL **")
    # common_data.save_model(model, MODEL_FILENAME)
    # test(args.path, saver)
    print("\t\t ** EXITING **")

if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
