# coding=utf-8
import os
import time
import numpy as np
from keras import layers
import keras.backend as K
from keras.models import load_model
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
from tools import get_mfcc_1,calculate_eer,get_mfcc_2
import constants as c
from keras.models import Model


'''
说话人确认
使用EER作为测试指标
'''


def t_loss(x):
	margin = 0.2
	return  K.relu(margin + x[0] - x[1])


def train_loss(y_true, y_pred):
	return y_pred


def test_model(model):
	num_layers = len(model.layers)  # 0-35
	x = np.random.randn(1, 13,300,3)
	outputs = []

	for i in range(num_layers):
		get_ith_layer_output = K.function([model.get_input_at(0), K.learning_phase()],  # get_input_at   layers[2].input
										  [model.layers[i].output])
		layer_output = get_ith_layer_output([x, 0])[0]  # output in test mode = 0
		outputs.append(layer_output)

	for i in range(num_layers):
		print("Shape of layer {} output:{}".format(i, outputs[i].shape))


def score(testfile,fa_data_dir,iden_model_load_path):
	print("Use {} for test".format(testfile))

	verify_list = np.loadtxt(testfile, str)

	verify_lb = np.array([int(i[0]) for i in verify_list])
	list1 = np.array([os.path.join(fa_data_dir, i[1]) for i in verify_list])
	list2 = np.array([os.path.join(fa_data_dir, i[2]) for i in verify_list])

	total_list = np.concatenate((list1, list2))
	unique_list = np.unique(total_list)

	model_name_list = os.listdir(iden_model_load_path)
	min_eer = 100
	min_model = ""
	for model_name in model_name_list:
		this_model_path = os.path.join(iden_model_load_path, model_name)
		print("============================")
		print("Load model form {}".format(this_model_path))
		# Load model
		model = load_model(this_model_path,custom_objects={"t_loss":t_loss,"train_loss":train_loss})
		# model.summary()
		# test_model(model)
		# exit(0)

		# 获取特征输出  dense_1
		output = model.get_layer("model_1").get_output_at(0)
		# output = model.get_layer("dense_1").get_output_at(0)
		model = Model(inputs=model.get_layer("model_1").get_input_at(0), outputs=output)
		# model.summary()
		# exit(0)

		print("Start testing...")
		total_length = len(unique_list)  # 4715
		feats, scores, labels = [], [], []
		for c, ID in enumerate(unique_list):
			if c % 100 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
			# specs = get_mfcc_2(ID)
			specs = get_mfcc_1(ID)
			v = model.predict(specs.reshape(1, *specs.shape))
			feats += [v]

		# print(feats[0].shape)
		# exit(0)
		feats = np.array(feats)

		# 计算余弦角度
		for c, (p1, p2) in enumerate(zip(list1, list2)):
			ind1 = np.where(unique_list == p1)[0][0]
			ind2 = np.where(unique_list == p2)[0][0]

			v1 = feats[ind1, 0]
			v2 = feats[ind2, 0]

			scores += [np.sum(v1 * v2)] # cos(o)
			labels += [verify_lb[c]]

		scores = np.array(scores)
		labels = np.array(labels)

		eer = calculate_eer(labels, scores,model_name)
		print("EER: {}".format(eer))
		if eer < min_eer:
			min_eer = eer
			min_model = model_name

	# 输出最终结果
	print("============================")
	print("Min_eer : ", min_eer)
	print("Model_name : ", min_model)
	print("============================")


if __name__ == '__main__':
	# VERI_TEST_FILE = "a_veri/veri_in_128.txt"
	VERI_TEST_FILE = "a_veri/veri_out.txt"
	# FA_DIR = "F:/vox_data_mfcc_npy/test_128/"
	FA_DIR = "F:/vox_data/vox1_dev_wav/wav/"
	VERI_MODEL_LOAD_PATH = "F:/models/veri/m_128"


	score(VERI_TEST_FILE,FA_DIR,VERI_MODEL_LOAD_PATH)