import time
import numpy as np
import constants as c
import keras
import tensorflow as tf
import keras.backend as K
from keras import layers
from keras.models import Model,Sequential,load_model
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers import Activation, Conv1D, Conv2D, Input, Lambda,Dropout,LSTM
from keras.layers import BatchNormalization, Flatten, Dense, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


def test_model(model):
	num_layers = len(model.layers)  # 0-35
	x = np.random.randn(1, 13,300,3)
	outputs = []

	for i in range(num_layers):
		get_ith_layer_output = K.function([model.layers[0].input, K.learning_phase()],
										  [model.layers[i].output])
		layer_output = get_ith_layer_output([x, 0])[0]  # output in test mode = 0
		outputs.append(layer_output)

	for i in range(num_layers):
		print("Shape of layer {} output:{}".format(i, outputs[i].shape))


'''
自定义卷积神经网络
输出形状的改变涉及：_gene_Data、identification(v = np.squeeze(v))
'''

def conNet(input_shape,weight_decay,pool):
	'''
	train: 0 ~ 30  0.001
		   35 ~ 85  0.0001
		   90 ~ 115  0.00006(2)
	'''
	# ===============================================
	#            input layer
	# ===============================================
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
	x3 = Conv2D(256, (4, 3),
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
	# x4 = Conv2D(256, (3, 3),
	# 			kernel_initializer='orthogonal',
	# 			use_bias=False, trainable=True,
	# 			kernel_regularizer=l2(weight_decay),
	# 			padding='same',
	# 			name='conv4_1')(x3)
	# x4 = Conv2D(384, (3, 3),
	# 			kernel_initializer='orthogonal',
	# 			use_bias=False, trainable=True,
	# 			kernel_regularizer=l2(weight_decay),
	# 			padding='same',
	# 			# strides=(2, 1),
	# 			name='conv4_2')(x4)
	# x4 = BatchNormalization(axis=3, epsilon=1e-5, momentum=1, name='bn4', trainable=True)(x4, training=False)
	# x4 = Activation('relu', name='relu4')(x4)
	# if pool == "max":
	# 	x4 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="mpool4")(x4)
	# elif pool == "avg":
	# 	x4 = AveragePooling2D((2, 2), strides=(2, 2), padding="same", name="avgpool4")(x4)
	# ===============================================
	#            GlobalAveragePooling
	# ===============================================
	a1 = GlobalAveragePooling2D(name='avg_pool')(x3)
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