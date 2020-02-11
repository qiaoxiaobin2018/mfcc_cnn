# 工具包
# 包括：获取 MFCC 特征、

import os
import time
import keras
import scipy.io.wavfile as wav
import constants as c
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import fbank

'''
读入训练文件列表 
FA_DIR：音频文件绝对路径前缀
path：训练文件列表
'''


def get_voxceleb1_datalist(FA_DIR, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(FA_DIR, string.split(",")[0]) for string in strings])
        labellist = np.array([int(string.split(",")[1]) for string in strings])
        f.close()
        audiolist = audiolist.flatten()
        labellist = labellist.flatten()
    return audiolist, labellist


'''
类：训练数据的生成器
'''


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, dim, max_sec, step_sec, frame_step, batch_size=2, n_classes=1251,
                 shuffle=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_sec = max_sec
        self.step_sec = step_sec
        self.frame_step = frame_step

        self.on_epoch_end()

    def __getitem__(self, index):
        '返回一个 batch_size 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        batch_data, batch_labels = self._gene_Data(list_IDs_temp, indexes)
        return batch_data, batch_labels

    def __len__(self):
        '计算有多少个 batch_size'
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def on_epoch_end(self):
        '每次迭代后打乱训练列表'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _gene_Data(self, list_IDs_temp, indexes):
        '得到 MFCC 特征向量数组和类标签，以输入模型进行训练'
        b_data = np.empty((self.batch_size,) + self.dim)
        b_labels = np.empty((self.batch_size,), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            b_data[i, :, :, :] = get_mfcc_1(ID)
            b_labels[i] = self.labels[indexes[i]]  # 0~n-1
            # b_labels[i] = self.labels[indexes[i]] - 1 # 1~n

        b_labels = keras.utils.to_categorical(b_labels, num_classes=self.n_classes) # (None,n_class)
        # b_labels = b_labels.reshape(self.batch_size, 1, 1, self.n_classes) # (None,1,1,n_class)
        # os.system("pause")

        return b_data, b_labels


'''
获取 MFCC 特征 
问题：标准化（转置之前还是之后）、语音截取、
'''


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


def get_mfcc(filepath):
    (rate, siglist) = wav.read(filepath)
    mfcc_feat = mfcc(siglist, c.SAMPLE_RATE, winlen=c.FRAME_LEN, winstep=c.FRAME_STEP, numcep=13, nfilt=26,
                     nfft=c.NUM_FFT)

    # d_mfcc_feat_1 = delta(mfcc_feat, 2)
    # d_mfcc_feat_2 = delta(d_mfcc_feat_1, 2)

    features = mfcc_feat # np.stack((mfcc_feat,d_mfcc_feat_1),axis=2)
    features = features[0:300, :]
    features_norm = normalize_frames(features.T)
    features = features_norm.T

    return features


def get_mfcc_1(filepath):
    (rate, siglist) = wav.read(filepath)
    mfcc_feat = mfcc(siglist, c.SAMPLE_RATE, winlen=c.FRAME_LEN, winstep=c.FRAME_STEP, numcep=13, nfilt=26,
                     nfft=c.NUM_FFT)

    length = mfcc_feat.shape[0]
    reserve_length = length - (length % 100)
    rstart = int((length - reserve_length) / 2)

    if c.MODE == "train":
        mfcc_feat = mfcc_feat[0:300, :]
    elif c.MODE == "test":
        mfcc_feat = mfcc_feat[0:reserve_length, :]

    d_mfcc_feat_1 = delta(mfcc_feat, 2)
    d_mfcc_feat_2 = delta(d_mfcc_feat_1, 2)

    mfcc_feat_normal = normalize_frames(mfcc_feat.T)
    d_mfcc_feat_1_normal = normalize_frames(d_mfcc_feat_1.T)
    d_mfcc_feat_2_normal = normalize_frames(d_mfcc_feat_2.T)

    features = np.stack((mfcc_feat_normal,d_mfcc_feat_1_normal,d_mfcc_feat_2_normal),axis=2)

    return features


'''绘制训练损失图像'''


def draw_loss_img(history_dict,save_path):
    loss_values = history_dict['loss']  # 训练损失
    # val_loss_values = history_dict['val_loss']  # 验证损失
    ep = range(1, len(loss_values) + 1)

    # plt.switch_backend('agg')
    plt.plot(ep, loss_values, 'b', label="Training loss")  # bo表示蓝色原点
    # plt.plot(ep, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


'''绘制训练精度图像'''


def draw_acc_img(history_dict,save_path):
    accs = history_dict['acc']  # 训练精度
    # val_acc = history_dict['val_acc']  # 验证精度
    ep = range(1, len(accs) + 1)

    # plt.switch_backend('agg')
    plt.plot(ep, accs, 'b', label="Training Acc")  # bo表示蓝色原点
    # plt.plot(ep, val_acc, 'b', label="Validation Acc")  # b表示蓝色实线
    plt.title("Train Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()  # 绘图
    plt.savefig(save_path)
    plt.show()


'''
测试
'''


if __name__ == '__main__':
    mfcc = get_mfcc_1("F:/vox_data/vox1_dev_wav/wav/id10001/1zcIwhmdeo4/00001.wav")
    print(mfcc.shape)
    # print(mfcc)