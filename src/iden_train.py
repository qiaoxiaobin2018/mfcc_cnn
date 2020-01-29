import os
import time
import keras
import tools
import constants as c
import keras.backend as K
import tensorflow as tf
from keras import optimizers
from keras.models import load_model
from model import conNet


def train_vggvox_model(model_load_path, model_save_path,continue_training, save_model):
    audiolist, labellist = tools.get_voxceleb1_datalist(c.FA_DIR, c.IDEN_TRAIN_LIST_FILE)
    train_gene = tools.DataGenerator(audiolist, labellist, c.DIM, c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP, c.BATCH_SIZE,
                                     c.N_CLASS)
    if continue_training == 1:
        print("load model from {}...".format(model_load_path))
        model = load_model(model_load_path, custom_objects={'tf': tf})
        print("pre LR: ", K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, c.LR)
        print("lat LR: ",K.get_value(model.optimizer.lr))
    else:
        model = conNet(c.INPUT_SHAPE,c.WEIGHT_DECAY,c.POOL)
        # 编译模型
        model.compile(optimizer=optimizers.Adam(lr=c.LR,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      loss="categorical_crossentropy",  # 使用分类交叉熵作为损失函数
                      metrics=['acc'])  # 使用精度作为指标
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(c.IDEN_MODEL_FA_PATH,'iden_model_64_{epoch:02d}_{loss:.3f}_{acc:.3f}_conNet_add30.h5'),
                                                 monitor='loss',
                                                 mode='min',
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 period=5)]

    print("Start training...")
    history = model.fit_generator(train_gene,
                                  epochs=c.EPOCHS,
                                  steps_per_epoch=int(len(labellist) // c.BATCH_SIZE),
                                  callbacks=callbacks
                                  )

    # print("save weights to {}...".format(c.PERSONAL_WEIGHT))
    # model.save_weights(filepath=c.PERSONAL_WEIGHT, overwrite=True)
    if save_model == 1:
        print("save model to {}...".format(model_save_path))
        model.save(model_save_path, overwrite=True)
    tools.draw_loss_img(history.history,c.LOSS_PNG)
    tools.draw_acc_img(history.history,c.ACC_PNG)
    print("Done!")

'''
训练
'''


if c.MODE == "test":
    print("must be train mode!")
    exit(0)
print("*****Check params*****\nmode:{}\nlearn_rate:{}\nepochs:{}\nbatch_size:{}\nclass_num:{}\ncontinue_training:{}\nsave_model:{}\n*****Check params*****"
      .format(c.MODE,c.LR,c.EPOCHS,c.BATCH_SIZE,c.N_CLASS,c.CONTINUE_TRAINING,c.SAVE))
time.sleep(10)
# set_learning_phase(0)
train_vggvox_model(c.IDEN_MODEL_PATH, c.IDEN_MODEL_PATH, c.CONTINUE_TRAINING, c.SAVE)
