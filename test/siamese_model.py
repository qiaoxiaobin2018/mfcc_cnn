import os
import time
import numpy as np
import constants as c
import keras
import tensorflow as tf
import keras.backend as K
from keras import layers,optimizers
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras.layers.merge import concatenate
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


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred): # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred): # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def get_voxceleb1_datalist(TRAIN_FA_DIR, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist1 = np.array([os.path.join(TRAIN_FA_DIR, string.split(",")[0]) for string in strings])
        audiolist2 = np.array([os.path.join(TRAIN_FA_DIR, string.split(",")[1]) for string in strings])
        labellist = np.array([int(string.split(",")[2]) for string in strings])
        f.close()
        audiolist1 = audiolist1.flatten()
        audiolist2 = audiolist2.flatten()
        labellist = labellist.flatten()
    return audiolist1, audiolist2,labellist


def get_mfcc_2(filepath):
    mfccc = np.load(filepath)
    return mfccc


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list1_IDs, list2_IDs,labels, dim, batch_size=2, n_classes=1251,shuffle=True):
        self.list1_IDs = list1_IDs
        self.list2_IDs = list2_IDs
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.on_epoch_end()

    def __getitem__(self, index):
        '返回一个 batch_size 的数据'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list1_IDs_temp = [self.list1_IDs[k] for k in indexes]
        list2_IDs_temp = [self.list2_IDs[k] for k in indexes]
        batch_data1,batch_data2,batch_labels = self._gene_Data(list1_IDs_temp,list2_IDs_temp,indexes)
        return [batch_data1, batch_data2],batch_labels

    def __len__(self):
        '计算有多少个 batch_size'
        return int(np.floor(len(self.list1_IDs)) / self.batch_size)

    def on_epoch_end(self):
        '每次迭代后打乱训练列表'
        self.indexes = np.arange(len(self.list1_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _gene_Data(self, list1_IDs_temp, list2_IDs_temp,indexes, b_data=None):
        '得到 mfcc 特征向量数组和类标签，以输入模型进行训练'
        b_data1 = np.empty((self.batch_size,) + self.dim)
        b_data2 = np.empty((self.batch_size,) + self.dim)
        b_labels = np.empty((self.batch_size,), dtype=int)
        for i, ID in enumerate(list1_IDs_temp):
            b_data1[i, :, :, :] = get_mfcc_2(ID)
            b_labels[i] = self.labels[indexes[i]]
        for i, ID in enumerate(list2_IDs_temp):
            b_data2[i, :, :, :] = get_mfcc_2(ID)

        # b_labels = keras.utils.to_categorical(b_labels, num_classes=self.n_classes) # (None,n_class)
        # b_labels = b_labels.reshape(self.batch_size, 1, 1, self.n_classes) # (None,1,1,n_class)
        # os.system("pause")

        return b_data1,b_data2,b_labels


def get_siamese_model(model_load_path):
    # 导入已有模型
    print("Load model form {}".format(model_load_path))
    base_model = load_model(model_load_path)
    # test_model(base_model)
    # exit(0)

    # 定义模型
    output = base_model.layers[15].output
    output = layers.Lambda(lambda x: K.l2_normalize(x, 1))(output)
    base_model = Model(inputs=base_model.layers[0].input, outputs=output)

    input_a = Input(shape=(13, 300, 3))
    input_b = Input(shape=(13, 300, 3))
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    # keras.utils.plot_model(model, "siamModel.png", show_shapes=True)
    # model.summary()

    return model


if __name__ == '__main__':
    # 常量
    train_fa_dir = "F:/vox_data_mfcc_npy/train_veri_128/"
    train_list = "D:/Python_projects/fbank_cnn/a_veri/train_128_pairs_for_veri.txt"

    model_load_path = "F:/models/iden/m_128/iden_model_64_0.8737.h5"
    dim = (13,300,3)

    LR = 0.001
    EPOCHS = 10
    BATCH_SIZE = 8
    N_CLASS = 2

    # 获取模型
    model = get_siamese_model(model_load_path)
    # 编译
    model.compile(loss=contrastive_loss,
                  optimizer=optimizers.Adam(lr=LR,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=[accuracy])
    # 准备数据
    audiolist1, audiolist2,labellist = get_voxceleb1_datalist(train_fa_dir,train_list)
    train_gene = DataGenerator(audiolist1,audiolist2, labellist, dim, BATCH_SIZE, N_CLASS)
    # 训练
    print("*****Check params*****\n"
          "learn_rate:{}\n"
          "epochs:{}\n"
          "batch_size:{}\n"
          "class_num:{}\n"
          "*****Check params*****"
        .format(LR, EPOCHS, BATCH_SIZE, N_CLASS))
    time.sleep(10)
    print("Start training...")
    history = model.fit_generator(train_gene,
                                  epochs=EPOCHS,
                                  steps_per_epoch=int(len(labellist) // c.BATCH_SIZE),
                                  # callbacks=callbacks
                                  )

