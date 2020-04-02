import os
import time
import numpy as np
# import constants as c
import keras
import tensorflow as tf
import keras.backend as K
from keras import layers,optimizers
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers.merge import dot
from keras.layers import Activation, Conv1D, Conv2D, Input, Lambda, Dropout, LSTM
from keras.layers import BatchNormalization, Flatten, Dense, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


def test_model(model):
	num_layers = len(model.layers)  # 0-35
	x = np.random.randn(1, 13, 300, 3)
	outputs = []

	for i in range(num_layers):
		get_ith_layer_output = K.function([model.layers[0].input, K.learning_phase()],
										  [model.layers[i].output])
		layer_output = get_ith_layer_output([x, 0])[0]  # output in test mode = 0
		outputs.append(layer_output)

	for i in range(num_layers):
		print("Shape of layer {} output:{}".format(i, outputs[i].shape))


def get_voxceleb1_datalist(TRAIN_FA_DIR, path):
	with open(path) as f:
		strings = f.readlines()
		owner = np.array([os.path.join(TRAIN_FA_DIR, string.split(",")[0]) for string in strings])
		plist = np.array([os.path.join(TRAIN_FA_DIR, string.split(",")[1]) for string in strings])
		nlist = np.array([os.path.join(TRAIN_FA_DIR, string.split(",")[2][0:-1]) for string in strings])
		f.close()
		owner = owner.flatten()
		plist = plist.flatten()
		nlist = nlist.flatten()
	return owner, plist,nlist


def get_mfcc_2(filepath):
	mfccc = np.load(filepath)
	return mfccc



class DataGenerator(keras.utils.Sequence):
	def __init__(self, olist, plist,nlist, dim, batch_size=2, n_classes=1251,shuffle=True):
		self.olist = olist
		self.plist = plist
		self.nlist = nlist
		self.dim = dim
		self.batch_size = batch_size
		self.n_classes = n_classes
		self.shuffle = shuffle

		self.on_epoch_end()

	def __getitem__(self, index):
		'返回一个 batch_size 的数据'
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		olist_temp = [self.olist[k] for k in indexes]
		plist_temp = [self.plist[k] for k in indexes]
		nlist_temp = [self.nlist[k] for k in indexes]
		batch_olist,batch_plist,batch_nlist = self._gene_Data(olist_temp,plist_temp,nlist_temp,indexes)

		b_labels = np.zeros((self.batch_size,), dtype=int)

		# return [batch_olist],[batch_plist],[batch_nlist]
		# return  {"input_1": batch_olist,
		# 		 "input_2": batch_plist,
		# 		 "input_3": batch_nlist}
		return [batch_olist,batch_plist,batch_nlist],b_labels



	def __len__(self):
		'计算有多少个 batch_size'
		return int(np.floor(len(self.olist)) / self.batch_size)

	def on_epoch_end(self):
		'每次迭代后打乱训练列表'
		self.indexes = np.arange(len(self.olist))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def _gene_Data(self, olist_temp,plist_temp,nlist_temp,indexes, b_data=None):
		'得到 mfcc 特征向量数组和类标签，以输入模型进行训练'
		olist_npy = np.empty((self.batch_size,) + self.dim)
		plist_npy = np.empty((self.batch_size,) + self.dim)
		nlist_npy = np.empty((self.batch_size,) + self.dim)
		for i, ID in enumerate(olist_temp):
			olist_npy[i, :, :, :] = get_mfcc_2(ID)
		for i, ID in enumerate(plist_temp):
			plist_npy[i, :, :, :] = get_mfcc_2(ID)
		for i, ID in enumerate(nlist_temp):
			nlist_npy[i, :, :, :] = get_mfcc_2(ID)

		# print(olist_npy.shape)
		# print(plist_npy.shape)
		# print(nlist_npy.shape)
		# exit(0)

		# olist_npy = np.array(olist_npy)
		# plist_npy = np.array(plist_npy)
		# nlist_npy = np.array(nlist_npy)

		return olist_npy,plist_npy,nlist_npy


def t_loss(x):
	margin = 0.1
	return  K.relu(margin + x[0] - x[1])



def get_triple_model(model_load_path):
	# 导入已有模型
	print("Load model form {}".format(model_load_path))
	base_model = load_model(model_load_path)
	# test_model(base_model)
	# exit(0)

	# 定义模型
	output = base_model.layers[15].output
	output = layers.Lambda(lambda x: K.l2_normalize(x, 1))(output)
	base_model = Model(inputs=base_model.layers[0].input, outputs=output)

	input_o = Input(shape=(13, None, 3),name="input_a")
	input_p = Input(shape=(13, None, 3),name="input_b")
	input_n = Input(shape=(13, None, 3),name="input_c")
	owner_feature = base_model(input_o)
	posi_feature = base_model(input_p)
	nega_feature = base_model(input_n)

	owner_feature = Dense(128)(owner_feature)  # 一般的做法是，直接讲问题和答案用同样的方法encode成向量后直接匹配，但我认为这是不合理的，我认为至少经过某个变换。

	right_cos = dot([owner_feature, posi_feature], -1, normalize=True)
	wrong_cos = dot([owner_feature, nega_feature], -1, normalize=True)

	# loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
	loss = Lambda(t_loss)([wrong_cos, right_cos])

	train_model = Model(inputs=[input_o, input_p, input_n], outputs=loss)
	model_o_encoder = Model(inputs=input_o, outputs=owner_feature)
	model_p_encoder = Model(inputs=input_p, outputs=posi_feature)

	# keras.utils.plot_model(train_model, "triplet_Model.png", show_shapes=True)
	# exit(0)
	# model.summary()
	# exit(0)

	return train_model,model_o_encoder,model_p_encoder


def train_loss(y_true, y_pred):
	return y_pred


if __name__ == '__main__':
	# 常量
	# ================windows===================
	train_fa_dir = "F:/vox_data_mfcc_npy/train_128_for_veri/"
	train_list = "D:/Python_projects/mfcc_cnn/a_veri/triple_train_64_in.txt"

	ori_model_load_path = "F:/models/veri/load/iden_model_64_0.8737.h5"
	con_model_load_path = "F:/models/veri/iden_model_test.h5"

	veri_model_save_fa_path = "F:/models/veri/m_128/"
	# ================linux===================
	# train_fa_dir = "/home/longfuhui/vox_data_mfcc_npy/train_128_for_veri/"  # /export/longfuhui/vox_mfcc_npy/train_128_for_veri
	# train_list = "/home/longfuhui/shengwenshibie/mfcc_cnn/a_veri/tmp.txt"
    #
	# ori_model_load_path = "/home/longfuhui/models/veri/load_model/iden_model_64_0.8496.h5"
	# con_model_load_path = "/home/longfuhui/models/veri/iden_model_test.h5"
	# veri_model_save_fa_path = "/home/longfuhui/models/veri/m_128/"
	# ==================consts================
	continue_training = 0
	LR = 0.001
	EPOCHS = 20
	BATCH_SIZE = 32
	period = 2
	N_CLASS = 2
	dim = (13, 300, 3)
	# ========================================
	train_model = model_o_encoder =  model_p_encoder = None
	# 获取模型
	if continue_training == 1:
		print("load model from {}...".format(con_model_load_path))
		train_model = load_model(con_model_load_path,custom_objects={"t_loss":t_loss,"train_loss":train_loss})
		print("pre LR: ", K.get_value(train_model.optimizer.lr))
		K.set_value(train_model.optimizer.lr, LR)
		print("lat LR: ",K.get_value(train_model.optimizer.lr))
	else:
		train_model,model_o_encoder,model_p_encoder = get_triple_model(ori_model_load_path)
		# 编译
		train_model.compile(loss= train_loss, # lambda y_true,y_pred: y_pred,
							optimizer=optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
		model_o_encoder.compile(loss="mse",
								optimizer='adam')
		model_p_encoder.compile(loss="mse",
								optimizer='adam')

	# 准备数据
	ownerlist, plist,nlist = get_voxceleb1_datalist(train_fa_dir,train_list)
	train_gene = DataGenerator(ownerlist,plist, nlist, dim, BATCH_SIZE, N_CLASS)
	# 训练
	print("*****Check params*****\n"
		  "learn_rate:{}\n"
		  "epochs:{}\n"
		  "period:{}\n"
		  "batch_size:{}\n"
		  "class_num:{}\n"
		  "*****Check params*****"
		.format(LR, EPOCHS, period,BATCH_SIZE, N_CLASS))
	time.sleep(5)
	print("Start training...")
	callbacks = [keras.callbacks.ModelCheckpoint(
		os.path.join(veri_model_save_fa_path, 'iden_model_64_{epoch:02d}_{loss:.3f}_conNet.h5'),
		monitor='loss',
		mode='min',
		save_best_only=True,
		save_weights_only=False,
		period=period)
		]
	history = train_model.fit_generator(train_gene,
								  epochs=EPOCHS,
								  steps_per_epoch=int(len(ownerlist) // BATCH_SIZE),
								  callbacks=callbacks
								  )
	# 最终保存模型
	print("save model to {}...".format(con_model_load_path))
	train_model.save(con_model_load_path, overwrite=True)
	# model_o_encoder.save(os.path.join(owner_model_save_path, 'owner_model_{}.h5'.format(add)))
	# model_p_encoder.save(os.path.join(posi_model_save_path, 'posi_model_{}.h5'.format(add)))

