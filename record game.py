import numpy as np
from mss import mss
from PIL import Image
from getKey import Key_check
import time
import os
import cv2
from threading import Thread


GAME_TOP = 84
GAME_LEFT = 898
GAME_WIDTH = 1021
GAME_HEIGHT =767

RECORD_WIDTH = 224
RECORD_HEIGHT = 224
IS_RGB = True

FILEINDEX = 2
FLODERNAME = "dataset"
FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"


def key2onehot(key):
    # " none u ul ur r l"

    out = [0 for i in range(6)]
    print(key)
    if ("up" in key):
        if ('left' in key):
            out[2] = 1
        elif ('right' in key):
            out[3] = 1
        else:
            out[1] = 1
    elif ('left' in key):
        out[5] = 1
    elif ('right' in key):
        out[4] = 1
    else:
        out[0] = 1

    return out

def save2npy(data):
    print(f"Saving data to {FILENAME}...")
    np.save(FILENAME,data)
    print(f"===== SUCCESSFUL =====")

def main():
    global FILENAME,FILEINDEX
    
    mon = {'top': GAME_TOP, 'left': GAME_LEFT, 'width': GAME_WIDTH, 'height': GAME_HEIGHT}
    sct = mss()

    while os.path.isfile(FILENAME):   
        FILEINDEX += 1
        FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"

    train_data = []
    print(f"opening new data file {FILENAME}")

    for i in range(5):
        print(f"start in {5-i}s")
        time.sleep(1)

    last_time = time.time()
    while True:

        sct.get_pixels(mon)
        img = np.array(Image.frombytes('RGB', (sct.width, sct.height), sct.image))

        if not IS_RGB:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(RECORD_WIDTH,RECORD_HEIGHT))

        key = Key_check()
        frame_choice = key2onehot(key)
        

        if ('p' in key or 'P' in key):
            print("---- pause ---- \n press \"C\" to continue\n press \"S\" to save data\n press \"N\" to create new file")
            print(" press \"D\" to clear saved data")
            while True:
                key = Key_check()
                if ('c' in key or 'C' in key):
                    for i in range(5):
                        print(f"start in {5-i}s")
                        time.sleep(1)
                    break
                if ('s' in key or 'S' in key):
                    FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
                    Thread(target = save2npy, args = (train_data,), daemon = True).start()

                if ('d' in key or 'D' in key):
                    train_data = []
                    if os.path.isfile(FILENAME):
                        os.remove(FILENAME)
                    print("delete saved data")
                if ('n' in key or 'N' in key):
                    FILEINDEX+= 1
                    FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
                    while os.path.isfile(FILENAME):   
                        FILEINDEX += 1
                        FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
                    train_data = []
                    print(f"opening new data file {FILENAME}")
                    time.sleep(1)
                key = []
                time.sleep(0.2)
        print("FPS = ",1/abs(last_time-time.time()),end=" ")
        print(frame_choice)
        last_time = time.time()

        train_data.append([img,frame_choice])
        if (len(train_data) % 250 ==0):
            FILENAME = f"{FLODERNAME}\\train_data{FILEINDEX}.npy"
            Thread(target = save2npy, args = (train_data,), daemon = True).start()

        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

if __name__=="__main__":
    main()

