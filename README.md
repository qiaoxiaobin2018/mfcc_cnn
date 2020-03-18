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
		* SET test model father path in IDEN_MODEL_LOAD_PATH
		* RUN identification.py

### 从已有模型继续训练：
		* MODE --> "train"
		* CONTINUE_TRAINING --> 1
		* Adjust learning rate (LR)
		* SET model load path in IDEN_MODEL_PATH
		* RUN iden_train.py
		* model saved in IDEN_MODEL_FA_PATH

### License
	MIT License

	Copyright (c) 2020 qiaoxiaobin2018

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.

