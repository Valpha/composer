from __future__ import print_function
import numpy as np
from keras.models import load_model
from keras.layers import Input, Add, Conv2D, Dense, ZeroPadding2D, Activation, MaxPooling2D, Flatten
from keras.layers import Dense
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
import os
import cv2
import os
import keras
from keras.layers import Input, Add, Conv2D, Dense, ZeroPadding2D, Activation, MaxPooling2D, Flatten
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = load_model('lenet-5.h5')
# images = []
# sourceDir = '/home/Valpha/workspace/data/test/'
# for file in os.listdir(sourceDir):
#     # 图片路径
#     imgPath = os.path.join(sourceDir, file)
#
#     # 读取图片
#     x = cv2.imread(os.path.expanduser(imgPath), 0)
#
#     # 图片预处理：
#     # 1.由于我的模型训练是将所有图片缩放为50x50，所以这里也对图片做同样操作
#     x = cv2.resize(x, dsize=(40, 20), interpolation=cv2.INTER_LINEAR)
#     x = x.astype(float)
#
#     # 2.模型训练时图片的处理操作（此处根据自己模型的实际情况处理）
#     x *= (1. / 255)
#
#     # 存入list
#     images.append(x)
#
# xx = np.array(images)
#
# # 开始预测
# result = model.predict(xx)
# # 打印结果
# print(file + "  -->  " + str(result))

images = []
sourceDir = '/home/Valpha/workspace/data/test/'
for file in os.listdir(sourceDir):
    file_path = os.path.join(sourceDir, file)
    img = image.load_img(file_path, target_size=(40, 20), color_mode='grayscale')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = Input(x)

    y = model.predict(x, batch_size=1, verbose=1)
    images.append(y)
print(images)
# print('Predicted:', decode_predictions(y, top=3)[0])
