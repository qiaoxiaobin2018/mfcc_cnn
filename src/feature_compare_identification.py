import os
import time
import numpy as np
import pandas as pd
from keras import layers
from keras.models import load_model
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
import constants as c
import keras.backend as K
from tools import get_mfcc_2
from keras.models import Model


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
	y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
	y_pred *= scale
	return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


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


def get_embeddings_from_list_file(model, list_file,file_fa_path):
	result = pd.read_csv(list_file, delimiter=",")
	# print(result)
	# exit(0)
	result['features'] = result['filename'].apply(lambda x: get_mfcc_2(os.path.join(file_fa_path,x)))
	# print(result['features'])
	# exit(0)
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape))))
	# print(result['embedding'])
	# exit(0)
	return result[['filename','speaker','embedding']]


'''
方法：输入同一个人的不同语音，计算输出向量之间的余弦夹角，得到评分
'''


def feature_compare_result(test_list,enroll_list,iden_model_load_path,enroll_file_fa_path,test_file_fa_path,result_file):
	# 读入测试数据、标签
	print("Use {} for test".format(test_list))

	model_name_list = os.listdir(iden_model_load_path)
	Top_acc = 0
	Top_model = ""
	for model_name in model_name_list:
		this_model_path = os.path.join(iden_model_load_path, model_name)
		print("============================")
		print(this_model_path)
		print("Load model form {}".format(this_model_path))
		model = load_model(this_model_path, custom_objects={'amsoftmax_loss': amsoftmax_loss})
		# test_model(model)
		# exit(0)

		# 获取特征输出
		output = model.layers[15].output
		output = layers.Lambda(lambda x: K.l2_normalize(x, 1))(output)
		model = Model(inputs=model.layers[0].input, outputs=output)

		# 处理注册语音
		print("Processing enroll samples....")
		enroll_result = get_embeddings_from_list_file(model, enroll_list, enroll_file_fa_path)
		enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
		print("{} voice enrolled!".format(len(enroll_embs)))
		# print(enroll_embs)
		# exit(0)
		speakers = enroll_result['speaker']
		# print("speakers: ",speakers.shape)# [3,]
		# print(speakers)# [19,13,27] three class

		print("Processing test samples....")
		test_result = get_embeddings_from_list_file(model, test_list, test_file_fa_path)
		test_embs = np.array([emb.tolist() for emb in test_result['embedding']])
		print("{} voice for test!".format(len(test_embs)))
		# print(test_embs)
		# exit(0)

		print("Comparing test samples against enroll samples....")
		distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric="cosine"), columns=speakers)  # 计算余弦相似度（ 夹角，越小越好 ）
		# print("distances: ")
		# print(distances)
		# exit(0)

		scores = pd.read_csv(test_list, delimiter=",", header=0, names=['test_file', 'test_speaker'])
		# scores = pd.concat([scores, distances],axis=1)
		# scores['result'] = scores[speakers].idxmin(axis=1)  # 取最小值第一次出现的索引

		scores['result'] = distances[speakers].idxmin(axis=1)  # 取最小值第一次出现的索引 （改动：仅输出结果，删除余弦相似度）
		scores['correct'] = (scores['result'] == scores['test_speaker'])  # bool to int
		res = scores['correct'].value_counts(1)
		print(res)
		acc = max(res)
		print("acc : ",acc)
		if(acc > Top_acc):
			Top_acc = acc
			Top_model = model_name
			# 保存模型
			print("Writing outputs to [{}]....".format(result_file))
			result_dir = os.path.dirname(result_file)
			if not os.path.exists(result_dir):
				os.makedirs(result_dir)
			with open(result_file, 'w') as f:
				scores.to_csv(f, index=False)
	# 输出最终结果
	print("============================")
	print("Top_acc : ", Top_acc)
	print("Model_name : ", Top_model)
	print("============================")


if __name__ == '__main__':
	voice_length = 3 # 6 9
	iden_model_load_path = "F:/models/iden/m_128/" # iden_model_128_22_0.666_0.991_conNet_add20.h5
	test_list = "D:/Python_projects/mfcc_cnn/a_iden/test_64.csv"
	if voice_length == 3:
		enroll_list = "D:/Python_projects/mfcc_cnn/a_iden/enroll_64_3s.csv"
		enroll_file_fa_path = "F:/vox_data_mfcc_npy/enroll_128_3-5s"
	elif voice_length == 6:
		enroll_list = "D:/Python_projects/mfcc_cnn/a_iden/enroll_64_6s.csv"
		enroll_file_fa_path = "F:/vox_data_mfcc_npy/enroll_128_6-8s"
	else:
		enroll_list = "D:/Python_projects/mfcc_cnn/a_iden/enroll_64_9s.csv"
		enroll_file_fa_path = "F:/vox_data_mfcc_npy/enroll_128_9-11s"
	test_file_fa_path = "F:/vox_data_mfcc_npy/test_128/"
	result_file = "D:/Python_projects/mfcc_cnn/a_iden/iden_compare_res.csv"

	time_start = time.time()
	feature_compare_result(test_list,enroll_list,iden_model_load_path,enroll_file_fa_path,test_file_fa_path,result_file)
	time_end = time.time()
	print('cost: ', time_end - time_start)
