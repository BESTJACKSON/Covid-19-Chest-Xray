# coding=utf-8
import cv2
#opencv的库
import os, shutil
import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import load_model
import numpy as np
import sys

font = cv2.FONT_HERSHEY_SIMPLEX
from keras.optimizers import Adam
import utils
from scipy import misc



CLASSES = (
'NORMAL','PNEUMONIA')

model = load_model('model/model-ResNet50-final.h5')

imgName = 'test/PNEUMONIA/person1_bacteria_1.jpg'
code = utils.ImageEncode(imgName)

ret = model.predict(code)
print(ret)
#Enter the category with the greatest similarity
res1 = np.argmax(ret[0, :])

print('result:', CLASSES[res1])

