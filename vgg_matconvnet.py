'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import LRN
import numpy as np
#from rotate_img import rotate_color_space

import os



def VGG():
    global mitjana

    # input image dimensions
    img_rows, img_cols = 224, 224
    num_classes = 1000

    import scipy.io
    mat = scipy.io.loadmat('imagenet-vgg-m.mat')


    def weights_meus(shape, dtype=None):
        return weights

    def bias_meus(shape, dtype=None):
        if kernelX == 1:
            return bias
        else:
            return bias.flatten()

    def bias_meus_fc(shape, dtype=None):
        if kernelX == 1:
            return bias.transpose().flatten()
        else:
            return bias.flatten()

    model = Sequential()
    # ACTIVATION ______________________________________(224x224x3)->(224x224x3)
    #    model.add(Activation('linear',input_shape=(img_rows, img_cols,3) ))



    #    #(1) CONV1 ______________________________________ (224x224x3)->(109x109x96)
    layer = 0
    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    nfilters = weights.shape[3]
    kernelX = weights.shape[0]
    kernelY = weights.shape[1]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]

    model.add(Conv2D(nfilters, kernel_size=(kernelX, kernelY),
                     strides=(stride, stride),
                     padding="same",
                     kernel_initializer=weights_meus,
                     bias_initializer=bias_meus,
                     input_shape=(img_rows, img_cols, 3)))

    crop = ((1, 2), (1, 2))
    model.add(Cropping2D(cropping=crop, data_format=None))

    # (2) RELU1 _____________________________________ (109x109x96)->(109x109x96)
    layer += 1
    model.add(Activation('relu'))

    # (3) LRN1 ______________________________________ (109x109x96)->(109x109x96)
    layer += 1
    param = mat['layers'][0][layer]['param'][0][0][0]
    N = param[0]
    kappa = param[1]
    alpha = param[2]
    beta = param[3]
    model.add(LRN.LRN2D(alpha=alpha, k=kappa, beta=beta, n=N))

    # (4) POOL1 ________________________________________(109x109x96)->(54x54x96)
    layer += 1
    poli = mat['layers'][0][layer]['pool'][0][0][0]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]
    model.add(MaxPooling2D(pool_size=(poli[0], poli[1]), padding="same", strides=(stride, stride)))

    crop = ((0, 1), (0, 1))
    model.add(Cropping2D(cropping=crop, data_format=None))

    # (5) CONV2 ________________________________________ (54x54x96)->(26x26x256)
    layer = layer + 1
    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    nfilters = weights.shape[3]
    kernelX = weights.shape[0]
    kernelY = weights.shape[1]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]

    model.add(Conv2D(nfilters, kernel_size=(kernelX, kernelY),
                     strides=(stride, stride),
                     padding="same",
                     kernel_initializer=weights_meus,
                     bias_initializer=bias_meus))

    crop = ((0, 1), (0, 1))
    model.add(Cropping2D(cropping=crop, data_format=None))

    # (6) RELU2 _______________________________________ (26x26x256)->(26x26x256)
    layer += 1
    model.add(Activation('relu'))

    # (7) LRN2 ________________________________________ (26x26x256)->(26x26x256)
    layer += 1
    param = mat['layers'][0][layer]['param'][0][0][0]
    N = param[0]
    kappa = param[1]
    alpha = param[2]
    beta = param[3]
    model.add(LRN.LRN2D(alpha=alpha, k=kappa, beta=beta, n=N))

    # (8) POOL2________________________________________ (26x26x256)->(13x13x256)
    layer += 1
    poli = mat['layers'][0][layer]['pool'][0][0][0]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]
    model.add(MaxPooling2D(pool_size=(poli[0], poli[1]), padding="same", strides=(stride, stride)))

    # (9) CONV3 _______________________________________ (13x13x256)->(13x13x512)
    layer = layer + 1
    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    nfilters = weights.shape[3]
    kernelX = weights.shape[0]
    kernelY = weights.shape[1]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]

    model.add(Conv2D(nfilters, kernel_size=(kernelX, kernelY),
                     strides=(stride, stride),
                     padding="same",
                     kernel_initializer=weights_meus,
                     bias_initializer=bias_meus))

    # (10) RELU3 ______________________________________ (13x13x512)->(13x13x512)
    layer += 1
    model.add(Activation('relu'))

    # (11) CONV4 _______________________________________ (13x13x512)->(13x13x512)
    layer = layer + 1
    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    nfilters = weights.shape[3]
    kernelX = weights.shape[0]
    kernelY = weights.shape[1]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]
    model.add(Conv2D(nfilters, kernel_size=(kernelX, kernelY),
                     strides=(stride, stride),
                     padding="same",
                     kernel_initializer=weights_meus,
                     bias_initializer=bias_meus))

    # (12) RELU4 ______________________________________ (13x13x512)->(13x13x512)
    layer += 1
    model.add(Activation('relu'))

    # (13) CONV5 ______________________________________ (13x13x512)->(13x13x512)
    layer = layer + 1
    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    nfilters = weights.shape[3]
    kernelX = weights.shape[0]
    kernelY = weights.shape[1]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]
    model.add(Conv2D(nfilters, kernel_size=(kernelX, kernelY),
                     strides=(stride, stride),
                     padding="same",
                     kernel_initializer=weights_meus,
                     bias_initializer=bias_meus))

    # (14) RELU5 ______________________________________ (13x13x512)->(13x13x512)
    layer += 1
    model.add(Activation('relu'))

    # (15) POOL5 _________________________________________(13x13x512)->(6x6x512)
    layer += 1
    poli = mat['layers'][0][layer]['pool'][0][0][0]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]
    model.add(MaxPooling2D(pool_size=(poli[0], poli[1]), padding="same", strides=(stride, stride)))

    crop = ((0, 1), (0, 1))
    model.add(Cropping2D(cropping=crop, data_format=None))

    # (16) FC6 ___________________________________________ (6x6x512)->(1x1x4096)
    layer = layer + 1
    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    nfilters = weights.shape[3]
    kernelX = weights.shape[0]
    kernelY = weights.shape[1]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]
    model.add(Conv2D(nfilters, kernel_size=(kernelX, kernelY),
                     strides=(stride, stride),
                     padding="valid",
                     kernel_initializer=weights_meus,
                     bias_initializer=bias_meus))

    # (17) RELU6 ________________________________________ (1x1x4096)->(1x1x4096)
    layer += 1
    model.add(Activation('relu'))

    # (18) FC7 ___________________________________________ (1x1x4096)->(1x14096)
    layer = layer + 1
    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    nfilters = weights.shape[3]
    kernelX = weights.shape[0]
    kernelY = weights.shape[1]
    stride = mat['layers'][0][layer]['stride'][0][0][0][0]
    model.add(Conv2D(nfilters, kernel_size=(kernelX, kernelY),
                     strides=(stride, stride),
                     padding="same",
                     kernel_initializer=weights_meus,
                     bias_initializer=bias_meus_fc))

    # (19) RELU7 ________________________________________ (1x1x4096)->(1x1x4096)
    layer += 1
    model.add(Activation('relu'))

    # (20) FC8 __________________________________________ (1x1x4096)->(1x1x1000)
    #    layer=layer+1
    #    weights = mat['layers'][0][layer]['weights'][0][0][0][0]
    #    bias = mat['layers'][0][layer]['weights'][0][0][0][1]
    #    nfilters=weights.shape[3]
    #    kernelX=weights.shape[0]
    #    kernelY=weights.shape[1]
    #    stride=mat['layers'][0][layer]['stride'][0][0][0][0]
    #    def weights_meus(shape, dtype=None):
    #        return weights
    #    def bias_meus(shape, dtype=None):
    #        if kernelX==1:
    #            return bias.flatten()
    #        else:
    #            return bias.flatten()
    #    model.add(Conv2D(nfilters, kernel_size=(kernelX,kernelY),
    #                          strides=(stride,stride),
    #                          padding="same",
    #                          kernel_initializer=weights_meus,
    #                          bias_initializer=bias_meus))
    #
    # (21) SOFTMAX  _________________________________________(1x1x4096)->(1000)
    model.add(Flatten())

    layer = layer + 1
    model.add(Dense(num_classes, activation='softmax'))

    return model


