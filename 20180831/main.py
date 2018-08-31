import argparse
import numpy as np
import os

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Load a database and train on it.")
    parser.add_argument("-p", "--path", help="Path to dataset", required=True)
    parser.add_argument("-vp", "--validation_percentage", type=float, default=0.25, help="Percentage of training set used as validation")
    return parser.parse_args(argv)

def main(args):
    # Load all file paths

    # Get class labels for training
    # Filters out blank values, removes "\n", splits by "/" and get last position of strip.
    class_names = [ filter(None, i.rstrip().rsplit("/"))[-1] for i in os.popen("ls -1d {}/train/*/".format(args.path)).readlines() ]

    # Get images for each class and split into training and validation
    dataset_paths = {}
    training_images_paths = {}
    validation_images_paths = {}
    images_counter = 0
    for idx in range(0, len(class_names)):
        # Get all
        class_images = [ i.rstrip() for i in os.popen("ls {}train/{}/* | sort -V".format(args.path, class_names[idx])).readlines() ]
        class_images_len = len(class_images)
        images_counter += class_images_len
        # Split into training and validation based on idx
        validation_idx = int(np.floor(class_images_len * args.validation_percentage))
        training_images_paths[idx] = class_images[:class_images_len - validation_idx - 1]
        validation_images_paths[idx] = class_images[class_images_len - validation_idx:]

    print(training_images_paths)

if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
