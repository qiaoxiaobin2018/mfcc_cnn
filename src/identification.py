import os
import time
import numpy as np
import constants as c
import keras.backend as K
import tensorflow as tf
from keras.models import load_model
from tools import get_mfcc, get_mfcc_1, get_mfcc_2,get_fbank
from model import conNet

'''
os.environ["MKL_NUM_THREADS"] = '12'
os.environ["NUMEXPR_NUM_THREADS"] = '12'
os.environ["OMP_NUM_THREADS"] = '12'
'''


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


'''
说话人识别
使用Acc作为指标
'''


def iden(testfile,fa_data_dir,iden_model,mode):
    # 读入测试数据、标签
    print("mode: ",mode)
    print("Use {} for test".format(testfile))

    iden_list = np.loadtxt(testfile, str,delimiter=",")

    labels = np.array([int(i[1]) for i in iden_list])
    voice_list = np.array([os.path.join(fa_data_dir, i[0]) for i in iden_list])

    # test all models
    # get all model path
    model_name_list = os.listdir(iden_model)
    # print(model_name_list)
    Top_acc = 0
    Top_model = ""
    for model_name in model_name_list:
        this_model_path = os.path.join(iden_model , model_name)
        print("============================")
        print(this_model_path)
        print("Load model form {}".format(this_model_path))
        model = load_model(this_model_path, custom_objects={'amsoftmax_loss': amsoftmax_loss})

        # print(model.summary())
        # exit(0)

        print("Start identifying...")
        total_length = len(voice_list)
        res, p_labels = [], []
        for c, ID in enumerate(voice_list):
            if c % 100 == 0: print('Finish identifying for {}/{}th wav.'.format(c, total_length))
            mfcc = get_mfcc_2(ID)
            v = model.predict(mfcc.reshape(1, *mfcc.shape))  # mfcc.reshape(1, *mfcc.shape, 1)
            v = np.squeeze(v)
            p_labels.append(np.argmax(v))

        p_labels = np.array(p_labels)
        compare = (labels == p_labels)
        counts = sum(compare == True)
        acc = counts / total_length
        print(acc)

        if acc > Top_acc:
            Top_acc = acc
            Top_model = model_name

    print("============================")
    print("Top_acc : ",Top_acc)
    print("Model_name : ",Top_model)
    print("============================")


if c.MODE == "train":
    print("must be test mode!")
    # exit(0)
iden(c.IDEN_TEST_FILE,c.TEST_FA_DIR,c.IDEN_MODEL_LOAD_PATH,c.MODE)