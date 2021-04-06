"""
@file : model.py
@author : s.aparajith@live.com
@date : 23.4.2021
@license : MIT
@details : contains the NN model. Main entry point of the project.
"""
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, BatchNormalization
import numpy as np
from keras.models import load_model
import utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from os.path import isfile
import readData


def networkModel():
    """
    @brief defines a keras model.
    @details a model inspired by the nvidia model for AI driving.
            https://arxiv.org/pdf/1604.07316v1.pdf
    @returns model in Keras
    """
    model = Sequential()
    # input shape is (70, 240, 3)
    model.add(Conv2D(filters=24,
                     kernel_size=5,
                     strides=(2, 2),
                     padding='VALID',
                     input_shape=(70, 240, 3),
                     activation='relu',
                     dtype=np.float32))
    # output shape is (33, 118, 24)
    model.add(BatchNormalization())
    model.add(Conv2D(filters=36,
                     kernel_size=5,
                     strides=(2, 2),
                     padding='VALID',
                     activation='relu'))
    # output shape is (15, 60, 36)
    model.add(BatchNormalization())
    model.add(Conv2D(filters=48,
                     kernel_size=5,
                     strides=(2, 2),
                     padding='VALID',
                     activation='relu'))
    # output shape is (6, 28, 48)
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64,
                     kernel_size=3,
                     strides=(1, 1),
                     padding='VALID',
                     activation='relu'))
    # output shape is (4, 26, 64)
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64,
                     kernel_size=3,
                     strides=(1, 1),
                     padding='VALID',
                     activation='relu'))
    # output shape is (2, 24, 64)
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1152, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    # regression output
    model.add(Dense(1))
    return model


def Analyze(modelHistory):
    """
    @brief summarize history for loss
    @param modelHistory Keras Model history
    """
    plt.figure()
    plt.plot(modelHistory.history['loss'])
    plt.plot(modelHistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(1, len(modelHistory.history['loss']) + 1, step=1))
    plt.legend(['train', 'validation'], loc='upper left')
    plt.ylim((0, 1))
    plt.show()


def Train(X_train,
          y_Train,
          epochs=5,
          val_split=0.25,
          batchSize=32):
    """
    @brief Incremental training pipeline.
    @details this pipeline will train a model 10 epochs each time it is run.
             It will save the model between runs to make it possible to
             train on a Nvidia GTX 1650-TI Mobile GPU
    @param X_train: training set
    @param y_Train:
    @param epochs: epochs to train
    @param val_split: percent of split in training data
    @param batchSize: Keras model batch_size
    @return None
    """
    if not isfile('model.h5'):
        print("training for the very first time")
        modelObj = networkModel()
        opt = Adam(learning_rate=0.01)
        modelObj.compile(loss='mse', optimizer=opt)
        modelHist = modelObj.fit(X_train,
                                 y_Train,
                                 validation_split=val_split,
                                 shuffle=True,
                                 epochs=epochs,
                                 batch_size=batchSize)
        Analyze(modelHist)
        modelObj.save('model.h5')
    else:
        new_model = load_model('model.h5')
        modelHist = new_model.fit(X_train,
                                  y_Train,
                                  validation_split=val_split,
                                  shuffle=True,
                                  epochs=epochs,
                                  batch_size=batchSize)
        Analyze(modelHist)
        new_model.save('model.h5')


# main entry point
if __name__ == "__main__":
    # play some images.
    # utils.playSome(X_train, 0)
    X_train, Y_Train = readData.getData(filePath='A:/driving_log.csv')
    plt.hist(Y_Train, bins=30)
    plt.show()
    Train(X_train, Y_Train)
