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
style.use("dark_background")
import cv2

GAME_TOP = 84
GAME_LEFT = 898
GAME_WIDTH = 1021
GAME_HEIGHT =767

MODEL_NAME = "models\\model_11.h5"

RECORD_WIDTH = 240
RECORD_HEIGHT = 160
IS_RGB = True
def main():
    mon = {'top': GAME_TOP, 'left': GAME_LEFT, 'width': GAME_WIDTH, 'height': GAME_HEIGHT}
    sct = mss()

    print("Loading model ...")
    m = load_model(MODEL_NAME)
    # get_3rd_layer_output = K.function([m.layers[0].input],
    #                               [m.layers[5].output])
    print("=== Success ===") 

    for i, var in enumerate(m.trainable_variables):
        print(m.trainable_variables[i].name)
    
    last_time = time.time()
    last = None

    try:
        while True:
            sct.get_pixels(mon)
            img = np.array(Image.frombytes('RGB', (sct.width, sct.height), sct.image))
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    except KeyboardInterrupt:
        cv2.destroyWindow('img')
        
    while True:

        sct.get_pixels(mon)
        img = np.array(Image.frombytes('RGB', (sct.width, sct.height), sct.image))

        #img = grab_screen((GAME_LEFT,GAME_TOP,GAME_WIDTH + GAME_LEFT + 1, GAME_HEIGHT+ GAME_TOP + 1))

        if not IS_RGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(RECORD_WIDTH,RECORD_HEIGHT))

        if not IS_RGB:
            img_pre = img.reshape((1,*img.shape,1))/255.0
        else:
            img_pre = img.reshape((1,*img.shape))/255.0

        key = m.predict(img_pre)[0] #* np.array([1,0.2,1,0.7,1,1])


        #  {0: 0, 1: 2667, 2: 568, 3: 1164, 4: 99, 5: 29}


        move = np.argmax(key)

        # pydirectinput.keyDown('w',_pause = False)

        ## " none u ul ur r l"
        # print(move)

        if (last != move or move == 0):
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            pydirectinput.keyUp('w')
            # pydirectinput.keyUp('d')

        if(move==1):
            pydirectinput.keyDown("w",_pause = False)
            print("|",end=" ")
            # time.sleep(0.3)
            # pydirectinput.keyUp('d')
        if (move==2):
            pydirectinput.keyDown("a",_pause = False)
            pydirectinput.keyDown("w",_pause = False)
            print("<-",end=" ")

        if (move==3):
            pydirectinput.keyDown("d",_pause = False)
            pydirectinput.keyDown("w",_pause = False)
            print("->",end=" ")
        if (move==4):
            pydirectinput.keyDown("d",_pause = False)
            print(">",end=" ")
        if (move==5):
            pydirectinput.keyDown("a",_pause = False)
            print("<",end=" ")
        print(key)

        # print(abs(last_time-time.time()))
        # last_time = time.time()

        last = move

        # m_output = np.array(get_3rd_layer_output(img_pre)[0][-1][:][:])
        # one_output = m_output[:,:,32]
        # new_output = one_output

        # for i in range(1,len(m_output[0][0])):
        #     new_output = np.concatenate((new_output, m_output[:,:,0]),axis = 1)
        # print(one_output.shape)
        # cla() 
        # imshow(one_output)
        # pause(0.1)wa

        # img_putpur = Image.fromarray(np.floor(one_output*255),'RGB')
        # img_putpur = img_putpur.resize((600, 600))
        # cv2.imshow('img',img)
        # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     break
    # show()
if __name__=="__main__":
    main()

