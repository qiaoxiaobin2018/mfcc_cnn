import os
import time
import numpy as np
import constants as c
import tensorflow as tf
from keras.models import load_model
from tools import get_mfcc,get_mfcc_1
from model import conNet

'''
os.environ["MKL_NUM_THREADS"] = '12'
os.environ["NUMEXPR_NUM_THREADS"] = '12'
os.environ["OMP_NUM_THREADS"] = '12'
'''


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

    # Load model
    print("Load model form {}".format(iden_model))
    model = load_model(iden_model, custom_objects={'tf': tf})

    print("Start identifying...")
    total_length = len(voice_list)
    res, p_labels = [], []
    for c, ID in enumerate(voice_list):
        if c % 100 == 0: print('Finish identifying for {}/{}th wav.'.format(c, total_length))
        mfcc = get_mfcc_1(ID)
        v = model.predict(mfcc.reshape(1, *mfcc.shape))  # mfcc.reshape(1, *mfcc.shape, 1)
        v = np.squeeze(v)
        p_labels.append(np.argmax(v))

    p_labels = np.array(p_labels)
    compare = (labels == p_labels)
    counts = sum(compare==True)
    acc = counts/total_length
    print(acc)


if c.MODE == "train":
    print("must be test mode!")
    exit(0)
iden(c.IDEN_TEST_FILE,c.FA_DIR,c.IDEN_MODEL_LOAD_PATH,c.MODE)