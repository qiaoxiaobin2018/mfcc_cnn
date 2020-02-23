# mfcc_cnn

环境：
pip 9.0.1
tensorflow-gpu 1.5.0
keras 2.1.4
cuda 9.0
cudnn 7.0.5

训练：
set train params in constants.py:
    CONTINUE_TRAINING --> 0
    LR = 0.001
    EPOCHS = 30
    BATCH_SIZE = 32
    N_CLASS = 128
mode --> "train"
run iden_train.py
model saved in IDEN_MODEL_FA_PATH

测试：
mode --> "test"
load model from IDEN_MODEL_LOAD_PATH
run identification.py

从已有模型继续训练：
CONTINUE_TRAINING --> 1
Adjust learning rate (LR)
load model from IDEN_MODEL_PATH
run iden_train.py
model saved in IDEN_MODEL_FA_PATH
