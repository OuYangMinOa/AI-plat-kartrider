#getKeys.py

import win32api as wapi
import time

left = 37
up = 38
right = 39
down = 40
shift = 16
ctrl = 17


KeyList=["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnqrstuvwxyz 123456789,.'APS$/\\":
    KeyList.append(char)
def Key_check():
        Keys = []
        for Key in KeyList:
            if wapi.GetAsyncKeyState(ord(Key)):
                Keys.append(Key)
        for Key in [left ,up, right, down, shift, ctrl]:
            if wapi.GetAsyncKeyState(Key):
                if Key==37:
                    Key="left"
                elif Key==38:
                    Key="up"
                elif Key==39:
                    Key="right"
                elif Key==40:
                    Key="down"
                elif Key == shift:
                    Key = "shift"
                elif Key == ctrl:
                    Key = "ctrl"
                Keys.append(Key)
        return Keys


        
if __name__=="__main__":
    while True:
        print(Key_check())
