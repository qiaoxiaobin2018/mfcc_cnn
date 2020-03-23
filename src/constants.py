
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


TEST_FA_DIR = "F:/vox_data_mfcc_npy/test_128/"   # /home/longfuhui/all_data/vox1-dev-wav/wav/
TRAIN_FA_DIR = "F:/vox_data_mfcc_npy/train_128/"  # /home/longfuhui/vox_data_mfcc_npy/train_128/
LOSS_PNG = "img/loss.png"
ACC_PNG = "img/acc.png"

'''
model compile setting
'''


CONTINUE_TRAINING = 1
SAVE = 1
LR = 0.00001
EPOCHS = 30
BATCH_SIZE = 32
N_CLASS = 128

'''
Identification
'''


MODE = "train"  # train or test



# train
IDEN_TRAIN_LIST_FILE = "a_iden/train_128_mfcc_npy.txt"  # /home/longfuhui/shengwenshibie/mfcc_cnn/a_iden/train_128_mfcc_npy.txt
IDEN_MODEL_FA_PATH = "F:/models/iden/m_128/" # /home/longfhui/models/iden/m_128/
IDEN_MODEL_PATH = "F:/models/iden/iden_model_test.h5" # /home/longfhui/models/iden/iden_model_test.h5

# test
IDEN_TEST_FILE = "a_iden/test_128.txt" # /home/longfuhui/shengwenshibie/mfcc_cnn/a_iden/test_128.txt
IDEN_MODEL_LOAD_PATH = "F:/models/iden/m_128" # /home/longfhui/models/iden/m_128/iden_model_128_20_0.094_1.000_conNet_add40.h5

