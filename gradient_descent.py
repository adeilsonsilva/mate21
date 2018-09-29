import argparse
import cv2
import json
import numpy as np
import os

MAX_EPOCHS = 5000
IMG_WIDTH = 71
IMG_HEIGHT = 77
L_RATE = 1e-3
MAX_DLOSS = 1e-5
BATCH_SIZE = 32
FILENAME = "gradient_descent.json"

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
            # Reshapes image to a 1D vector and converts all pixels to [0, 1]
            image = image.reshape(IMG_WIDTH*IMG_HEIGHT) / 255.0
            training_images.append([ image, idx ])

        # Load validation images
        for image_path in class_images[class_images_len - validation_idx:]:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Reshapes image to a 1D vector and converts all pixels to [0, 1]
            image = image.reshape(IMG_WIDTH*IMG_HEIGHT) / 255.0
            validation_images.append([ image, idx ])

    return training_images, validation_images

def linear_regression(x, w, b):
    y = np.dot(x, w) + b
    return y

def gradient_descent_step(b0, w0, images, learning_rate=L_RATE):
    # compute gradients
    b_grad = 0
    w_grad = 0
    N = len(images)
    for i in range(N):
        x = images[i][0]
        y = images[i][1]
        # Compute linear regression with initial weight/bias
        y_predicted = linear_regression(x, w0, b0)
        # w_grad is the derivative of the MSE in the direction of the weights
        w_grad += (2.0/N)*x*(y_predicted - y)
        # b_grad is the derivative of the MSE in the direction of the bias
        b_grad += (2.0/N)*(y_predicted - y)

    # print("\t\t*** GRADIENTS: b_grad={} ; w_grad={}".format(b_grad, w_grad))
    # update parameters
    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)

    return b1, w1

def loss_function(bias, weights, images):
    MSE = 0
    ACC = 0
    N = len(images)
    for i in range(N):
        x = images[i][0]
        y = images[i][1]
        y_predicted = linear_regression(x, weights, bias)
        # print("\t\t\t ŷ:{} y:{}".format(y_predicted, y))
        MSE += (1/N)*(((y_predicted - y)**2))
        if y == int(np.floor(y_predicted)):
            ACC += (1/N)
    return MSE, ACC

def train(training_images, validation_images, learning_rate=L_RATE, max_epochs=MAX_EPOCHS):
    # Generate random weights and bias
    bias = 0
    weights = np.random.uniform(low=-1e-1, high=1e-1, size=(IMG_WIDTH*IMG_HEIGHT,))
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
        bias, weights = gradient_descent_step(bias, weights, training_images[0:BATCH_SIZE])
        loss, accuracy = loss_function(bias, weights, validation_images)

        # Check for best accuracy
        print("+++ ACC: {}".format(accuracy))
        if accuracy >= max_acc:
            max_acc = accuracy
            max_acc_bias = bias
            max_acc_weights = weights
            max_acc_loss = loss
            max_acc_epoch = epoch

        # Check convergence
        print("--- LOSS: {}".format(loss))
        d_loss = abs(loss - prev_loss)
        if d_loss <= MAX_DLOSS:
            print("\t\t ** CONVERGED!")
            break
        prev_loss = loss

    if epoch >= max_epochs-1:
        print("\t\t %% MAX EPOCHS REACHED!! %%")

    model = {
        'last': {'acc': accuracy, 'bias': bias, 'weights': weights.tolist(), 'loss': loss, 'epoch': epoch},
        'best': {'acc': max_acc, 'bias': max_acc_bias, 'weights': max_acc_weights.tolist(), 'loss': max_acc_loss, 'epoch': max_acc_epoch}
    }

    return model

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

def main(args):
    print("\t\t ** LOADING IMAGES **")
    training_images, validation_images = load_images(args.path, args.validation_percentage)
    print("\t\t ** TRAINING MODEL **")
    model = train(training_images, validation_images)
    print("\t\t ** SAVING MODEL **")
    save_model(model)

if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
