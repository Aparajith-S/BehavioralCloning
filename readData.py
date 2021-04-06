"""
@file : readData.py
@author : s.aparajith@live.com
@date : 20.3.2021
@license : MIT
@details : contains functions to read the data.
"""
import numpy as np
from cv2 import imread
import csv
import tensorflow as tf
import preprocessing
import transforms

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9))
config.gpu_options.allow_growth = True
print(tf.config.list_physical_devices('GPU'))


def getData(filePath):
    """
    @brief function to get the training data. Please
    @param filePath path to csv file from unity project.
    @returns augmented training Data and labels.
    """
    lines = []
    with open(filePath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    X_train = []
    measurements = []
    count = 0
    for line in lines:
        source_path = line[0:3]
        # filename = source_path.split('/')[-1]
        measurement = float(line[3])
        for path in source_path:
            image = imread(path)
            # if steering angle is 0
            if 0.00001 > measurement > -0.00001:
                # replace with a randomized value between -0.1 and +0.1
                randomSteer = np.random.random() * 0.01 - 0.005
                # take every 15th value with 0.0 as steer angle
                if count % 15 == 0:
                    measurements.append(measurement + randomSteer)
                    X_train.append(preprocessing.preprocess(image))
                count = count + 1
            else:
                # Limit model from applying full steering.
                if measurement > 0.9:
                    measurement = 0.9
                if measurement < - 0.9:
                    measurement = - 0.9
                measurements.append(measurement)
                # transform the image and augment
                # augmentation is done only for track images with curves.
                proc = preprocessing.preprocess(image)
                X_train.append(proc)
                aug = []
                if -0.4 > measurement or measurement > 0.30:
                    aug = transforms.augmentData(image, 1)
                if -0.9 >= measurement or measurement > 0.50:
                    aug += transforms.augmentData(image, 1)
                if -0.6 > measurement > -0.9:
                    aug += transforms.augmentData(image, 1)
                # append augmented data into training set.
                for im in aug:
                    proc = preprocessing.preprocess(im)
                    X_train.append(proc)
                    measurements.append(measurement)
    X_train = np.array(X_train)
    y_Train = np.array(measurements)
    return X_train, y_Train


def getSimpleData(filePath):
    """
    @brief function to get the training data. Please
    @param filePath path to csv file from unity project.
    @returns training Data and labels that are not augmented/corrected by any means
    """
    lines = []
    with open(filePath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    X_train = []
    measurements = []
    count = 0
    for line in lines:
        source_path = line[0:3]
        measurement = float(line[3])
        for path in source_path:
            image = imread(path)
            measurements.append(measurement)
            X_train.append(preprocessing.preprocess(image))
    X_train = np.array(X_train)
    y_Train = np.array(measurements)
    return X_train, y_Train
