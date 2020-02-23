mfcc_cnn
=====

### 环境：
		Anaconda 3-5.1.0 (Python 3.6)
		pip 9.0.1
		tensorflow-gpu 1.5.0
		keras 2.1.4
		cuda 9.0
		cudnn 7.0.5

### 训练：
		* SET train params in constants.py:
			CONTINUE_TRAINING --> 0
			LR = 0.001
			EPOCHS = 30
			BATCH_SIZE = 32
			N_CLASS = 128
		* MODE --> "train"
		* RUN iden_train.py
		* model saved in IDEN_MODEL_FA_PATH

### 测试：
		* MODE --> "test"
		* SET test model in IDEN_MODEL_LOAD_PATH
		* RUN identification.py

### 从已有模型继续训练：
		* MODE --> "train"
		* CONTINUE_TRAINING --> 1
		* Adjust learning rate (LR)
		* SET model load path in IDEN_MODEL_PATH
		* RUN iden_train.py
		* model saved in IDEN_MODEL_FA_PATH
