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
		model = load_model(this_model_path, custom_objects={'amsoftmax_loss': amsoftmax_loss})
		# model.summary()
		# test_model(model)
		# time.sleep(5)
		# exit(0)

		# 获取特征输出
		output = model.layers[15].output
		output = layers.Lambda(lambda x: K.l2_normalize(x, 1))(output)
		model = Model(inputs=model.layers[0].input, outputs=output)
		print("Start testing...")
		total_length = len(unique_list)  # 4715
		feats, scores, labels = [], [], []
		for c, ID in enumerate(unique_list):
			if c % 200 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
			specs = get_mfcc_2(ID)
			# specs = get_mfcc_1(ID)
			v = model.predict(specs.reshape(1, *specs.shape))
			feats += [v]

		feats = np.array(feats)

		# 计算余弦角度
		for c, (p1, p2) in enumerate(zip(list1, list2)):
			ind1 = np.where(unique_list == p1)[0][0]
			ind2 = np.where(unique_list == p2)[0][0]

			v1 = feats[ind1, 0]
			v2 = feats[ind2, 0]

			# print(feats.shape)
			# print(v1.shape)

			# print(np.linalg.norm(v1))
			# print(np.linalg.norm(v2))

			scores += [np.sum(v1 * v2)] # cos(o)
			labels += [verify_lb[c]]

			# print(scores)
			# exit(0)

		# print("scores: ",scores)
		# print("labels: ",labels)

		# if c>0:
		#     break

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


def amsoftmax_loss(y_true, y_pred, scale=c.SCALE, margin=c.MARGIN):
	y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
	y_pred *= scale
	return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


if __name__ == '__main__':

	VERI_TEST_FILE = "a_veri/veri_in_128.txt"
	# VERI_TEST_FILE = "a_veri/veri_out.txt"
	FA_DIR = "F:/vox_data_mfcc_npy/test_128/"
	# FA_DIR = "F:/vox_data/vox1_dev_wav/wav/"
	VERI_MODEL_LOAD_PATH = "F:/models/veri/m_128/"


	score(VERI_TEST_FILE,FA_DIR,VERI_MODEL_LOAD_PATH)