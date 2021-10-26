#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 2021

@author: hejuxing
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

from keras import backend as K
if K.backend() == 'tensorflow':
    import tensorflow
    #K.set_image_dim_ordering('tf')
else:
    import theano
    #K.set_image_dim_ordering('th')

'''Ideally we should have changed image dim ordering based on Theano or Tensorflow, but for some reason I get following error when I switch it to 'tf' for Tensorflow.
	However, the outcome of the prediction doesnt seem to get affected due to this and Tensorflow gives me similar result as Theano.
	I didnt spend much time on this behavior, but if someone has answer to this then please do comment and let me know.
    ValueError: Negative dimension size caused by subtracting 3 from 1 for 'conv2d_1/convolution' (op: 'Conv2D') with input shapes: [?,1,200,200], [3,3,200,32].
'''
#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')
	
	
import numpy as np
#import matplotlib.pyplot as plt
import os

from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

import cv2
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# input image dimensions
img_rows, img_cols = 198, 198

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 3


# Batch_size to train
batch_size = 32

## Number of output classes (change it accordingly)
## If you change this then dont forget to change Labels accordingly
nb_classes = 5

# Number of epochs to train (change it accordingly)
nb_epoch = 25 

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

#%%
#  data
path = "./"

## Path1 is the folder which is fed in to training model
path1 = './train'
path2 = './test'

WeightFileName = []

# outputs
output = ["A0", "A1", "Else", "S0", "S1"]

jsonarray = {}



#%% For debug trace
def debugme():
    import pdb
    pdb.set_trace()


#%%
def modlistdir(path, pattern = None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if pattern == None:
            if name.startswith('.'):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)
            
    return retlist


# Load CNN model
def loadCNN(bTraining = False):
    global get_output
    model = Sequential()
    
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    # Model summary
    model.summary()
    # Model conig details
    model.get_config()
    
    if not bTraining :
        #List all the weight files available in current directory
        WeightFileName = modlistdir('.','.hdf5')
        if len(WeightFileName) == 0:
            print('Error: No pretrained weight file found. Please either train the model or download one from the https://github.com/asingh33/CNNGestureRecognizer')
            return 0
        else:
            print('Found these weight files - {}'.format(WeightFileName))
        #Load pretrained weights
        w = int(input("Which weight file to load (enter the INDEX of it, which starts from 0): "))
        fname = WeightFileName[int(w)]
        print("loading ", fname)
        model.load_weights(fname)

    # refer the last layer here
    layer = model.layers[-1]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    
    
    return model

# This function does the guessing work based on input images
def guessGesture(model):
    global output, get_output, jsonarray
    #Load image and flatten it
    imlist = modlistdir(path2)
    total_img = len(imlist)
    for n in range(len(imlist)):
            img=Image.open(path2 +'/' + imlist[n])
            img1=img.resize((img_rows,img_cols))
            image = np.array(img1).flatten()
    
            # reshape it
            image = image.reshape(img_channels, img_rows,img_cols)
    
            # float32
            image = image.astype('float32') 
    
            # normalize it
            image = image / 255
    
            # reshape for NN
            rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
            # Now feed it to the NN, to fetch the predictions
            index = model.predict_classes(rimage)
            prob_array = model.predict_proba(rimage)
    
            prob_array = get_output([rimage, 0])[0]
            print('prob_array: ',prob_array)
    
            d = {}
            i = 0
            for items in output:
                    d[items] = prob_array[0][i] * 100
                    i += 1
    
             # Get the output with maximum probability
            import operator
    
            guess = max(d.items(), key=operator.itemgetter(1))[0]
            prob  = d[guess]

            if prob > 25.0:
                print(guess, "  Probability: ", prob)
                with open('./modes.txt','a') as outfile:
                        outfile.write(guess + '   probability   ' + str(prob) + '\n')

            else:
                   # Lets return index 1 for 'Nothing' 
                return 1
            
  
    
    
#%%
def initializers():
    imlist = modlistdir(path1)
    img=Image.open(path1 +'/' + imlist[0])
    img1=img.resize((img_rows,img_cols))
    image1 = np.array(img1) # open one image to get size
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array((Image.open(path1+ '/' + images)).resize((img_rows,img_cols))).flatten()
                         for images in sorted(imlist)], dtype = 'f')
    

    
    print(immatrix.shape)
    
    input("Press any key")
    
    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    
    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''
    
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
     
    # Split X and y into training and testing sets
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0],img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0],img_channels, img_rows, img_cols)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train /= 255
    X_test /= 255
     
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test



def trainModel(model):

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname,overwrite=True)
    else:
        model.save_weights("newWeight.hdf5",overwrite=True)
        
    visualizeHis(hist)

    # Save model as well
    # model.save("newModel.hdf5")
#%%

def visualizeHis(hist):
    # visualizing losses and accuracy
    keylist = hist.history.keys()
    #print(hist.history.keys())
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    
    #Tensorflow new updates seem to have different key name
    if 'acc' in keylist:
        train_acc=hist.history['acc']
        val_acc=hist.history['val_acc']
    else:
        train_acc=hist.history['accuracy']
        val_acc=hist.history['val_accuracy']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    #plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()





