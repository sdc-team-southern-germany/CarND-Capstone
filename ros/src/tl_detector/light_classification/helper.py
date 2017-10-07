#!/usr/bin/python

import os
import numpy as np
#import cv2
from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imresize
import math
import random
from glob import glob
import re

# Image dimensions taken from squeezenet
WIDTH  = 227
HEIGHT = 227
# Ratio of training to test data [0:1]
RATIO  = 0.8

batch_size = 64


def get_class(label):
    '''
    :param new_label: Is the class index
    '''
    if label == "'stop'":
        new_label = 0                 #'red'
    elif label == "'warning'":
        new_label = 1                 #'yellow'
    elif label == "'go'":
        new_label = 2                 #'green'
    elif label == "'ambiguous'":
        new_label = 3                 #'unknown'
    return new_label

def get_image(image_file):
    #image = cv2.imread(image_file)
    image = imread(image_file, mode='RGB')
    image = resize_image(image)
    return image

def resize_image(image):
    #image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_LINEAR)
    image = imresize(image, (WIDTH, HEIGHT))
    return image

def gen_batch_function_LARA(data_path):
    """
    Generate function to create batches of training data
    :param data_path: Path to folder that contains all the datasets and labels

    ################
    # LARA Dataset #
    ################

    """
    # Paths to relative files
    #image_paths = glob(os.path.join(data_path, 'Lara3D_UrbanSeq1_JPG/*.jpg'))
    #labels_path = os.path.join(data_path, 'LARA_labels.txt')
    image_paths = glob(os.path.join(data_path, 'all_dataset')+"/*")
    labels_path = os.path.join(data_path, 'all_label.txt')
    # Read in label information
    label_no    = np.loadtxt(labels_path, dtype=int, delimiter=' ', skiprows=1, usecols=(1,))
    label_class = np.loadtxt(labels_path, dtype=str, delimiter=' ', skiprows=1, usecols=(2,))

    # Split labels into train and test
    l = len(label_no)
    indices = np.arange(l)
    random.shuffle(indices)
    train_indices = indices[0:int(l*RATIO)]
    test_indices  = indices[int(l*RATIO)+1:l]

    def get_batches_fn():
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        # Shuffle training data for each epoch
        random.shuffle(train_indices)

        for batch_i in range(0, len(train_indices), batch_size):
            images = []
            labels = []

            for index in train_indices[batch_i:batch_i+batch_size]:
                label = (label_class[index])
                labels.append(label)
                for image_file in image_paths:
                    pattern=".*_0*"+str(label_no[index])+"\."
                    if re.match(pattern, image_file):
                        images.append(get_image(image_file))
                        break

            # To visualise gen data
#            print(labels[0])
#            imshow(images[0])
#            cv2.imshow("Image window", images[0])
#            cv2.waitKey(5000)

            # Augment images
            images, labels = flip_lr(images, labels)
            # Yield
            yield np.array(images), np.array(labels)


    # Get X_test and y_test
    print('Generating test set... {}% train, {}% testing'.format(RATIO*100, (1-RATIO)*100))
    X_test = []
    y_test = []

    for index in test_indices:
        label = (label_class[index])
        y_test.append(label)

        for image_file in image_paths:
            pattern=".*_0*"+str(label_no[index])+"\."
            if re.match(pattern, image_file):
                X_test.append(get_image(image_file))
                break
    return get_batches_fn, np.array(X_test), np.array(y_test)

def calc_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def flip_lr(X, y):
    for i in range(len(y)):
        X.append(np.fliplr(X[i]))
        y.append(y[i])

    return X, y