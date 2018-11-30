import tensorflow as tf
import numpy as np
import cv2
import sys
import os

import common_data

model = './output/model.ckpt'

argc = len(sys.argv)
if (argc < 3):
    print("Usage: 'python test.py <test-path> <output-path> [model-path]'")
    sys.exit(-1)
elif (argc == 4):
    model = sys.argv[3]

path = sys.argv[1]
output = sys.argv[2]

print("\t\t** PATHS **")
print("\t+ Loading images from => '{}'".format(path))
print("\t+ Loading model from => '{}'".format(model))
print("\t+ Saving results to => '{}'".format(output))

# Evaluate model on test set
print("\t\t ** TESTING **")
paths = [ i.rstrip() for i in os.popen("ls {}test/* | sort -V".format(path)).readlines() ]

graph = tf.get_default_graph()
with tf.Session(graph=graph) as session:
    # model saver
    saver = tf.train.import_meta_graph(model + '.meta')
    saver.restore(session, model)

    tf_X = graph.get_tensor_by_name('net_input:0')
    tf_isTraining = graph.get_tensor_by_name('is_training:0')
    tf_result = graph.get_tensor_by_name('net_output:0')
    # a = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # print(a)

    f = open(output, "w")
    for image_path in paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(common_data.IMG_HEIGHT, common_data.IMG_WIDTH, common_data.NUM_CHANNELS) / 255.0

        y_predicted = session.run([tf_result], feed_dict = {tf_X: [image], tf_isTraining: False})

        # Get only image name (e.g. '14734.png')
        img_name = image_path.rstrip().rsplit("/")[-1]
        f.write("{} {}\n".format(img_name, y_predicted[0][0]))

    f.close()

print("\t\t ** EXITING **")
