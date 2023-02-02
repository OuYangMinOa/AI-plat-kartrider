import time
import pyautogui
import pydirectinput
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from mss import mss
from PIL import Image
from tensorflow.python.keras import backend as K
from matplotlib.pyplot import *
import cv2


RECORD_WIDTH = 160
RECORD_HEIGHT = 120
IS_RGB = True

FILENAME = "validation_data.npy"
MODEL_NAME = "models\\model_8_4.h5"

data = np.load(FILENAME,allow_pickle=True)

print("Loading model ...")
m = load_model(MODEL_NAME)
# get_3rd_layer_output = K.function([m.layers[0].input],
#                               [m.layers[5].output])

answer_choice = {}
predict_choice = {}
for i in range(6):
	answer_choice[i] = 0
	predict_choice[i] = 0
print("=== Success ===")

if (data[0].shape != ((RECORD_WIDTH,RECORD_HEIGHT))):
    print("reshaping the data ...")
    x = np.array([cv2.resize(each[0],(RECORD_WIDTH,RECORD_HEIGHT)) for each in list(data)])/255.0
    y = np.array([each[1] for each in list(data)])

else:
    x = np.array([each[0] for each in list(data)])/255.0
    y = np.array([each[1] for each in list(data)])

print(y)
for i in y:
	answer_choice[np.argmax(i)] += 1

print(answer_choice)



result = m.predict(x)
choices = np.argmax(result,axis=1)

for i in choices:
	predict_choice[i] += 1

print(predict_choice)


