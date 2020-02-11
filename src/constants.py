
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10
DIM = (13,300,3)

'''
model define
'''


WEIGHT_DECAY = 1e-4
INPUT_SHAPE = (13,None,3)
POOL = "max"  # max or avg

'''
IO on windows
'''


FA_DIR = "F:/Vox_data/vox1_dev_wav/wav/"
LOSS_PNG = "img/loss.png"
ACC_PNG = "img/acc.png"

'''
model compile setting
'''


CONTINUE_TRAINING = 0
SAVE = 1
LR = 0.001
EPOCHS = 30
BATCH_SIZE = 32
N_CLASS = 64

'''
Identification
'''


MODE = "train"  # train or test



# train
IDEN_TRAIN_LIST_FILE = "a_iden/tmp.txt"
IDEN_MODEL_FA_PATH = "F:/models/iden/m_128/"
IDEN_MODEL_PATH = "F:/models/iden/iden_model_test.h5" # iden_model_test.h5

# test
IDEN_TEST_FILE = "a_iden/test_tmp.txt"
IDEN_MODEL_LOAD_PATH = "F:/models/iden/m_128/iden_model_64_60_0.105_1.000_conNet_add30.h5"

