import time
import numpy as np
import constants as c
import keras
import tensorflow as tf
import keras.backend as K
from keras import layers
from keras.models import Model,Sequential
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers import Activation, Conv1D, Conv2D, Input, Lambda,Dropout,LSTM
from keras.layers import BatchNormalization, Flatten, Dense, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


'''
自定义卷积神经网络
输出形状的改变涉及：_gene_Data、identification(v = np.squeeze(v))
'''

def identity_block_2D(input_tensor, kernel_size, filters, stage, block, trainable=True,weight_decay = c.WEIGHT_DECAY):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis,  epsilon=1e-5, momentum=1, trainable=trainable, name=bn_name_1)(x, training=False)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis,  epsilon=1e-5, momentum=1, trainable=trainable, name=bn_name_2)(x, training=False)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis,  epsilon=1e-5, momentum=1, trainable=trainable, name=bn_name_3)(x, training=False)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_2D(input_tensor, kernel_size, filters, stage, block, strides=(1,1), trainable=True,weight_decay = c.WEIGHT_DECAY):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               # strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, trainable=trainable, name=bn_name_1)(x, training=False)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, trainable=trainable, name=bn_name_2)(x, training=False)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, trainable=trainable, name=bn_name_3)(x, training=False)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1),
                      strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      trainable=trainable,
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, trainable=trainable, name=bn_name_4)(shortcut, training=False)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def conNet(input_shape,weight_decay,pool):
    bn_axis = 3
    # ===============================================
    #            Input Layer
    # ===============================================
    inputs = Input(shape=input_shape, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(32, (3,3),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1/3x3_s1')(inputs)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, name='conv1_1/3x3_s1/bn', trainable=True)(x1, training=False)
    x1 = Activation('relu',name='relu1')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2),padding="same")(x1)
    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', trainable=True)
    # x2 = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, name='bn2', trainable=True)(x2, training=False)
    # x2 = Activation('relu', name='relu2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool2")(x2)
    # x2 = identity_block_2D(x2, 3, [16, 16, 64], stage=2, block='b', trainable=True)
    # x2 = identity_block_2D(x2, 3, [16, 16, 64], stage=2, block='c', trainable=True)
    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(x2, 3, [96, 96, 192], stage=3, block='a', trainable=True)
    # x3 = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, name='bn3', trainable=True)(x3, training=False)
    # x3 = Activation('relu', name='relu3')(x3)
    # x3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool3")(x3)
    # x3 = identity_block_2D(x3, 3, [32, 32, 128], stage=3, block='b', trainable=True)
    # x3 = identity_block_2D(x3, 3, [32, 32, 128], stage=3, block='c', trainable=True)
    # x3 = identity_block_2D(x3, 3, [32, 32, 128], stage=3, block='d', trainable=True)
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(x3, 3, [192, 192, 256], stage=4, block='a', strides=(2,1),trainable=True)
    # x4 = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
    # x4 = Activation('relu', name='relu4')(x4)
    x4 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool4")(x4)
    # x4 = identity_block_2D(x4, 3, [64, 64, 256], stage=4, block='b', trainable=True)
    # x4 = identity_block_2D(x4, 3, [64, 64, 256], stage=4, block='c', trainable=True)
    # x4 = identity_block_2D(x4, 3, [64, 64, 256], stage=4, block='d', trainable=True)
    # x4 = identity_block_2D(x4, 3, [64, 64, 256], stage=4, block='e', trainable=True)
    # x4 = identity_block_2D(x4, 3, [64, 64, 256], stage=4, block='f', trainable=True)
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    # x5 = conv_block_2D(x4, 3, [128, 128, 512], stage=5, block='a', trainable=True)
    # x5 = identity_block_2D(x5, 3, [128, 128, 512], stage=5, block='b', trainable=True)
    # x5 = identity_block_2D(x5, 3, [128, 128, 512], stage=5, block='c', trainable=True)
    # y = MaxPooling2D((2, 2), strides=(1, 2), name='mpool2')(x5)
    # ===============================================
    #            GlobalAveragePooling
    # ===============================================
    a1 = GlobalAveragePooling2D(name='avg_pool')(x4)
    # a2 = Reshape((1, 1, 128), name='reshape')(a1)
    # ===============================================
    #            Dense layer
    # ===============================================
    dense1 = Dense(256, activation='relu',
                   kernel_initializer='orthogonal',
                   use_bias=True, trainable=True,
                   kernel_regularizer=l2(weight_decay),
                   bias_regularizer=l2(weight_decay),
                   name='fc1')(a1)
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


'''
测试
'''


def test_for_cnn():
    input_shape = (13,300,3)
    model = conNet(input_shape,c.WEIGHT_DECAY,c.POOL)

    print(model.summary()) # 5,237,536  16,760,192
    exit(0)

    x = np.random.randn(1, 13,300,3)
    v = model.predict(x)
    print(v.shape)
    print("res:\n", v)


if __name__ == '__main__':
    test_for_cnn()