#Created on Wed Sep 8 2021

#@author: hejuxing


#package import
import cv2
import numpy as np
import os
import time

import threading

import modeCNN as myNN

banner =  '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition
    2- Train the model (you will require image samples for training under .\training)
    3- Exit	
    '''

def Main():
    global guessGesture, visualize, mod, gestname, path


        
    #Call CNN model loading callback
    while True:
        ans = int(input( banner))
        if ans == 1:
            mod = myNN.loadCNN()
            myNN.guessGesture(mod)
            input("Press any key to quit")
            break
        elif ans == 2:
            mod = myNN.loadCNN(True)
            myNN.trainModel(mod)
            input("Press any key to quit")
            break
        
        else:
            print("Get out of here!!!")
            return 0



if __name__ == "__main__":
    
    Main()



