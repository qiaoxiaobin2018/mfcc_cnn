# import os
# import time
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def draw(x,y,model_name):
#
#     plt.figure()
#     plt.plot(x,y, marker='*')
#     plt.xlim([0.0, 1.05])
#     plt.ylim([0.03, 0.13])
#     plt.xlabel('margin')
#     plt.ylabel('eer')
#     plt.title("eer_margin_figure_for_{}".format(model_name))
#     plt.legend()
#     plt.savefig(os.path.join("D:/Python_projects/mfcc_cnn/test/img", "eer_margin.png"))
#     plt.show()
#
#
# '''
# 执行
# '''
#
#
# if __name__ == '__main__':
#     x = np.array([0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.3,0.4,0.6,1.0])
#     y = np.array([0.0438,0.0415,0.0378,0.0429,0.0390,0.0431,0.0422,0.0478,0.0525,0.0595,0.1293])
#
#     if(len(x) != len(y)):
#         print("坐标点数不对应！")
#         exit(0)
#
#     draw(x,y,"Verification_2")





import wave

import pylab as pl

import numpy as np

# 打开WAV文档

f = wave.open(r"F:\vox_data\vox1_dev_wav\wav\id10001\1zcIwhmdeo4/00003.wav", "rb")

# 读取格式信息

# (nchannels, sampwidth, framerate, nframes, comptype, compname)

params = f.getparams()

nchannels, sampwidth, framerate, nframes = params[:4]

# 读取波形数据

str_data = f.readframes(nframes)

f.close()

#将波形数据转换为数组

wave_data = np.fromstring(str_data, dtype=np.short)

if nchannels == 2:

    wave_data.shape = -1, 2

    wave_data = wave_data.T

    time = np.arange(0, nframes) * (1.0 / framerate)

    # 绘制波形

    pl.subplot(211)

    pl.plot(time, wave_data[0])

    pl.subplot(212)

    pl.plot(time, wave_data[1], c="g")

    pl.xlabel("time (seconds)")

    pl.show()

elif nchannels == 1:

    wave_data.shape = -1, 1

    wave_data = wave_data.T

    time = np.arange(0, nframes) * (1.0 / framerate)

    # 绘制波形

    pl.subplot(211)

    pl.plot(time, wave_data[0])

    pl.xlabel("time (seconds)")

    pl.show()









