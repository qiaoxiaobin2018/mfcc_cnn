import time
import numpy as np
import constants as c
import keras.backend as K
import tensorflow as tf
from keras import layers
from keras.models import Model,Sequential
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers import Activation, Conv1D, Conv2D, Input, Lambda,Dropout,LSTM
from keras.layers import BatchNormalization, Flatten, Dense, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

'''
自定义卷积神经网络
'''


def trans(x):
    return tf.transpose(x, perm=[0, 2, 1])


def conNet_LSTM_2(input_shape,weight_decay,pool):
    # ===============================================
    # train: 0 ~ 30  0.001
    #        35 ~ 85  0.0001
    #        90 ~ 115  0.00006(2)
    # ===============================================
    # ===============================================
    #            input layer
    # ===============================================
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1_1
    # ===============================================
    x1_1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1_1')(inputs)
    x1_1 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1_1', trainable=True)(x1_1, training=False)
    x1_1 = Activation('relu', name='relu1_1')(x1_1)
    if pool == "max":
        x1_1 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool1_1")(x1_1)
    elif pool == "avg":
        x1_1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1_1")(x1_1)
    # ===============================================
    #            Convolution Block 1_2
    # ===============================================
    # x1_2 = Conv2D(16, (1,1),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             name='conv1_2_1')(inputs)
    # x1_2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1_2', trainable=True)(x1_2, training=False)
    # x1_2 = Activation('relu', name='relu1_2')(x1_2)
    # if pool == "max":
    #     x1_2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1_2")(x1_2)
    # elif pool == "avg":
    #     x1_2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1_2")(x1_2)
    # ===============================================
    #            Convolution Block 1_link
    # ===============================================
    # x1 = concatenate([x1_1, x1_2], axis=-1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1_1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(96, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv3_1')(x2)
    x3 = Conv2D(128, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2, 1),
                name='conv3_2')(x3)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    # a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
    a2 = Reshape((38, 128), name='reshape')(x3)
    # transpose = Lambda(trans)(a2)
    # ===============================================
    #            Dense layer
    # ===============================================
    # dense1 = Dense(128, activation='relu',
    #            kernel_initializer='orthogonal',
    #            use_bias=True, trainable=True,
    #            kernel_regularizer=l2(weight_decay),
    #            bias_regularizer=l2(weight_decay),
    #            name='fc1')(a1)
    # ===============================================
    #            LSTM layer
    # ===============================================
    lstm = LSTM(128,dropout=0.2,recurrent_dropout=0.2,name = "lstm_1")(a2)
    lstm = BatchNormalization(name='bn_lstm', trainable=True)(lstm, training=False)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(lstm)

    m = Model(inputs, s1, name='conNet')
    return m


def conNet_LSTM_1(input_shape,weight_decay,pool):
    # ===============================================
    # train: 0 ~ 30  0.001
    #        35 ~ 85  0.0001
    #        90 ~ 115  0.00006(2)
    # ===============================================
    # ===============================================
    #            input layer
    # ===============================================
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1_1
    # ===============================================
    x1_1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1_1')(inputs)
    x1_1 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1_1', trainable=True)(x1_1, training=False)
    x1_1 = Activation('relu', name='relu1_1')(x1_1)
    if pool == "max":
        x1_1 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool1_1")(x1_1)
    elif pool == "avg":
        x1_1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1_1")(x1_1)
    # ===============================================
    #            Convolution Block 1_2
    # ===============================================
    # x1_2 = Conv2D(16, (1,1),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             name='conv1_2_1')(inputs)
    # x1_2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1_2', trainable=True)(x1_2, training=False)
    # x1_2 = Activation('relu', name='relu1_2')(x1_2)
    # if pool == "max":
    #     x1_2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1_2")(x1_2)
    # elif pool == "avg":
    #     x1_2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1_2")(x1_2)
    # ===============================================
    #            Convolution Block 1_link
    # ===============================================
    # x1 = concatenate([x1_1, x1_2], axis=-1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1_1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(96, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv3_1')(x2)
    x3 = Conv2D(128, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2, 1),
                name='conv3_2')(x3)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    # a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
    a2 = Reshape((38, 128), name='reshape')(x3)
    # transpose = Lambda(trans)(a2)
    # ===============================================
    #            Dense layer
    # ===============================================
    # dense1 = Dense(128, activation='relu',
    #            kernel_initializer='orthogonal',
    #            use_bias=True, trainable=True,
    #            kernel_regularizer=l2(weight_decay),
    #            bias_regularizer=l2(weight_decay),
    #            name='fc1')(a1)
    # ===============================================
    #            LSTM layer
    # ===============================================
    lstm = LSTM(128,Dropout=0.2,recurrent_dropout=0.2,name = "lstm_1")(a2)
    lstm = BatchNormalization(name='bn_lstm', trainable=True)(lstm, training=False)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(lstm)

    m = Model(inputs, s1, name='conNet')
    return m


def conNet_f_vgg(input_shape,weight_decay,pool):
    # input_shape = (13,300,1)
    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(64,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1')(inputs)
    x1 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1, training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(128, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(1,2),
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block link
    # ===============================================
    # x3 = Conv2D(256, (3,3),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             name='conv3_1')(x2)
    # x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    # x3 = Activation('relu', name='relu3')(x3)
    # x4 = Conv2D(128, (3, 3),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             name='conv3_2')(x3)
    # x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    # x4 = Activation('relu', name='relu4')(x4)
    # if pool == "max":
    #     x4 = MaxPooling2D((2,2), strides=(1,2), padding="same", name="mpool4")(x4)
    # elif pool == "avg":
    #     x4 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            Convolution Block dynamic
    # ===============================================
    # x5 = Conv2D(128, (7,1),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='valid',
    #             name='conv5')(x4)
    # x5 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn5', trainable=True)(x5, training=False)
    # x5 = Activation('relu', name='relu5')(x5)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    # a1 = GlobalAveragePooling2D(name='avg_pool')(x5)
    # ===============================================
    #            Dense layer
    # ===============================================
    # dense1 = Dense(128, activation='relu',
    #            kernel_initializer='orthogonal',
    #            use_bias=True, trainable=True,
    #            kernel_regularizer=l2(weight_decay),
    #            bias_regularizer=l2(weight_decay),
    #            name='fc1')(a1)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.2)(dense1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    # s1 = Dense(c.N_CLASS, activation='softmax',
    #            kernel_initializer='orthogonal',
    #            use_bias=False, trainable=True,
    #            kernel_regularizer=l2(weight_decay),
    #            bias_regularizer=l2(weight_decay),
    #            name='prediction')(dense1)

    m = Model(inputs, x2, name='conNet')
    return m


def conNet_74_inception_module(input_shape,weight_decay,pool):
    # ===============================================
    # train: 0 ~ 30  0.001
    #        35 ~ 85  0.0001
    #        90 ~ 115  0.00006(2)
    # ===============================================
    # ===============================================
    #            input layer
    # ===============================================
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1_1
    # ===============================================
    x1_1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1_1')(inputs)
    x1_1 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1_1', trainable=True)(x1_1, training=False)
    x1_1 = Activation('relu', name='relu1_1')(x1_1)
    if pool == "max":
        x1_1 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool1_1")(x1_1)
    elif pool == "avg":
        x1_1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1_1")(x1_1)
    # ===============================================
    #            Convolution Block 1_2
    # ===============================================
    x1_2 = Conv2D(16, (1,1),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_2_1')(inputs)
    x1_2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1_2', trainable=True)(x1_2, training=False)
    x1_2 = Activation('relu', name='relu1_2')(x1_2)
    if pool == "max":
        x1_2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1_2")(x1_2)
    elif pool == "avg":
        x1_2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1_2")(x1_2)
    # ===============================================
    #            Convolution Block 1_link
    # ===============================================
    x1 = concatenate([x1_1, x1_2], axis=-1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(96, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv3_1')(x2)
    x3 = Conv2D(128, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2, 1),
                name='conv3_2')(x3)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
    # a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    dense1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a1)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.2)(dense1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(dense1)

    m = Model(inputs, s1, name='conNet')
    return m


def conNet_79_13_300_3(input_shape,weight_decay,pool):
    # ===============================================
    # train: 0 ~ 30  0.001
    #        35 ~ 85  0.0001
    #        90 ~ 115  0.00006(2)
    # ===============================================
    # ===============================================
    #            input layer
    # ===============================================
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1')(inputs)
    x1 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1, training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(96, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv3_1')(x2)
    x3 = Conv2D(128, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2, 1),
                name='conv3_2')(x3)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    # x4 = Conv2D(128, (1,3),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             strides=(1,2),
    #             name='conv4')(x3)
    # x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    # x4 = Activation('relu', name='relu4')(x4)
    # if pool == "max":
    #     x4 = MaxPooling2D((1,2), strides=(1,2), padding="same", name="mpool4")(x4)
    # elif pool == "avg":
    #     x4 = AveragePooling2D((1, 2), strides=(1, 2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
    a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    dense1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.2)(dense1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(dense1)

    m = Model(inputs, s1, name='conNet')
    return m


def conNet_75_13_300_3(input_shape,weight_decay,pool):
    # input_shape = (13,300,1)
    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1')(inputs)
    x1 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1, training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(128, (4, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2,1),
                name='conv3')(x2)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    # x4 = Conv2D(128, (1,3),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             strides=(1,2),
    #             name='conv4')(x3)
    # x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    # x4 = Activation('relu', name='relu4')(x4)
    # if pool == "max":
    #     x4 = MaxPooling2D((1,2), strides=(1,2), padding="same", name="mpool4")(x4)
    # elif pool == "avg":
    #     x4 = AveragePooling2D((1, 2), strides=(1, 2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
    a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    dense1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.2)(dense1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(dense1)

    m = Model(inputs, s1, name='conNet')
    return m


def conNet_69_13_300_1(input_shape,weight_decay,pool):

    # 转置之后 归一化

    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1')(inputs)
    x1 = BatchNormalization(axis=3,epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1,training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2, 2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(128, (4, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2, 1),
                name='conv3')(x2)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    # x4 = Conv2D(128, (1,6),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             strides=(1,2),
    #             name='conv4')(x3)
    # x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    # x4 = Activation('relu', name='relu4')(x4)
    # if pool == "max":
    #     x4 = MaxPooling2D((1, 2), strides=(1, 2), padding="same", name="mpool4")(x4)
    # elif pool == "avg":
    #     x4 = AveragePooling2D((1, 2), strides=(1, 2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
    a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    d1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.5)(d1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(d1)



    m = Model(inputs, s1, name='conNet')
    return m


def conNet_65_13_300_1(input_shape,weight_decay,pool):
    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1')(inputs)
    x1 = BatchNormalization(axis=3,epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1,training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2, 2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(128, (4, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2, 2),
                name='conv3')(x2)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    # x4 = Conv2D(128, (1,6),
    #             kernel_initializer='orthogonal',
    #             use_bias=False, trainable=True,
    #             kernel_regularizer=l2(weight_decay),
    #             padding='same',
    #             strides=(1,2),
    #             name='conv4')(x3)
    # x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    # x4 = Activation('relu', name='relu4')(x4)
    # if pool == "max":
    #     x4 = MaxPooling2D((1, 2), strides=(1, 2), padding="same", name="mpool4")(x4)
    # elif pool == "avg":
    #     x4 = AveragePooling2D((1, 2), strides=(1, 2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
    a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    d1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.5)(d1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(d1)



    m = Model(inputs, s1, name='conNet')
    return m


def conNet_0_300_13_N(input_shape,weight_decay,pool):
    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(32,(5,5),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1')(inputs)
    x1 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1, training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((1,2), strides=(1,2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(64, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(1,2),
                name='conv3')(x2)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    x4 = Conv2D(128, (1,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(1,2),
                name='conv4')(x3)
    x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    x4 = Activation('relu', name='relu4')(x4)
    if pool == "max":
        x4 = MaxPooling2D((1,2), strides=(1,2), padding="same", name="mpool4")(x4)
    elif pool == "avg":
        x4 = AveragePooling2D((1, 2), strides=(1, 2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x4)
    a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    dense1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.2)(dense1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(dense1)

    m = Model(inputs, x1, name='conNet')
    return m


def conNet_26_300_1(input_shape,weight_decay,pool):
    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(32,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1')(inputs)
    x1 = BatchNormalization(axis=3,epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1,training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2, 2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(64, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(2, 2),
                name='conv3')(x2)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    x4 = Conv2D(128, (2,4),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                strides=(1,2),
                name='conv4')(x3)
    x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    x4 = Activation('relu', name='relu4')(x4)
    if pool == "max":
        x4 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool4")(x4)
    elif pool == "avg":
        x4 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x4)
    a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    d1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.5)(d1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(d1)



    m = Model(inputs, s1, name='conNet')
    return m


def conNet_26_300_0(input_shape,weight_decay,pool):
    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(64,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1')(inputs)
    x1 = BatchNormalization(axis=3,epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1,training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(128, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2, 2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(128, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv3')(x2)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    x4 = Conv2D(128, (3,3),
                strides=(2,2),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv4')(x3)
    x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    x4 = Activation('relu', name='relu4')(x4)
    if pool == "max":
        x4 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool4")(x4)
    elif pool == "avg":
        x4 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x4)
    a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    d1 = Dense(128, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.5)(d1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(d1)



    m = Model(inputs, s1, name='conNet')
    return m


def conNet_40_300(input_shape,weight_decay,pool):
    # input layer
    inputs = Input(input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(64,(3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1')(inputs)
    x1 = BatchNormalization(axis=3,epsilon=1e-5, momentum=1, name='bn1', trainable=True)(x1,training=False)
    x1 = Activation('relu', name='relu1')(x1)
    if pool == "max":
        x1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool1")(x1)
    elif pool == "avg":
        x1 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool1")(x1)
    # ===============================================
    #            Convolution Block 2
    # ===============================================
    x2 = Conv2D(64, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv2')(x1)
    x2 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    x2 = Activation('relu', name='relu2')(x2)
    if pool == "max":
        x2 = MaxPooling2D((2,2), strides=(2, 2), padding="same", name="mpool2")(x2)
    elif pool == "avg":
        x2 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool2")(x2)
    # ===============================================
    #            Convolution Block 3
    # ===============================================
    x3 = Conv2D(128, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv3')(x2)
    x3 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    x3 = Activation('relu', name='relu3')(x3)
    if pool == "max":
        x3 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool3")(x3)
    elif pool == "avg":
        x3 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool3")(x3)
    # ===============================================
    #            Convolution Block 4
    # ===============================================
    x4 = Conv2D(128, (3,3),
                strides=(2,2),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv4')(x3)
    x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    x4 = Activation('relu', name='relu4')(x4)
    if pool == "max":
        x4 = MaxPooling2D((2,2), strides=(2,2), padding="same", name="mpool4")(x4)
    elif pool == "avg":
        x4 = AveragePooling2D((2,2), strides=(2,2), padding="same", name="avgpool4")(x4)
    # ===============================================
    #            Convolution Block 5
    # ===============================================
    x5 = Conv2D(256, (3, 3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv5')(x4)
    x5 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn5', trainable=True)(x5, training=False)
    x5 = Activation('relu', name='relu5')(x5)
    if pool == "max":
        x5 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool5")(x5)
    elif pool == "avg":
        x5 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool5")(x5)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x5)
    a2 = Reshape((1, 1, 256), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    d1 = Dense(256, activation='relu',
               kernel_initializer='orthogonal',
               use_bias=True, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='fc1')(a2)
    # ===============================================
    #            Dropout layer
    # ===============================================
    # d2 = Dropout(0.5)(d1)
    # ===============================================
    #            Softmax layer
    # ===============================================
    s1 = Dense(c.N_CLASS, activation='softmax',
               kernel_initializer='orthogonal',
               use_bias=False, trainable=True,
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay),
               name='prediction')(d1)



    m = Model(inputs, s1, name='conNet')
    return m


'''
测试
'''


def test():
    input_shape = (13,300,3)
    model = conNet_79_13_300_3(input_shape,c.WEIGHT_DECAY,c.POOL)

    print(model.summary()) # 5,237,536  16,760,192
    time.sleep(300000)

    x = np.random.randn(1, 16,256,1)
    print("x: ",x)
    v = model.predict(x)
    print(v.shape)
    print("res:\n", v)


if __name__ == '__main__':
    test()