#!/usr/bin/env python
# coding: utf-8



#Importing libraries

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import keras
import h5py
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, MaxPool2D, Input, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam
import os
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Imported Libraries')

# In[2]:


# #Unzip the dataset
# member = ('../GarbageClassification/garbage-classification.zip')
# from zipfile import ZipFile
# with ZipFile(member, 'r') as zipObj:
#        # Extract all the contents of zip file in current directory
#     zipObj.extractall()
 


# In[3]:

print('Reading the data')
#Reading the data
member = ('../GarbageClassification/Garbage classification/Garbage classification/')
catagories = os.listdir(member)
list_items = []
for cat in catagories:
    catagory_img = (member  + cat)
    #catagory_img.glob('*.jpeg')
    for _ in (glob.glob(catagory_img +'/'+'*.jpg')):
        list_items.append([cat, _])
    
#Convert list into dataframe

data = pd.DataFrame(list_items,columns = ['catagory', 'filepath'], index = None)
data = data.sample(frac=1).reset_index(drop=True)
data.head(5)
data.shape


# In[5]:


print('Splitting the dataset')
train_data = data[1:2000]
val_data = data[2001:2200]
test_data = data[2201:2527]




#-----------------*****************************************--------------------#

def adv_preprocessing(image):
    #loading images
    #Getting 3 images to work with
    preimgs = []
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    #Setting dimensions to resize
    height = 224
    width = 224
    
    dim = (width, height)
    res = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    preimgs.append(res)
        
#Removing noise from image - Gaussian blur
    
    blurred_img = cv2.GaussianBlur(res, (5,5),0)
    preimgs.append(blurred_img)

    #Segmentation 
    #------------------------------------------------------------------
    image = res
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    
    #More noise removal
    #------------------------------------------------------------------
    kernal = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal, iterations=2)
    
    #Sure background area
    sure_bg = cv2.dilate(opening, kernal, iterations = 3)
    
    #Finding foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    #Seperating different objects with different backgrounds
    #Markers labelling
    ret, markers  = cv2.connectedComponents(sure_fg)
    #Add one to all labels so that sure background is 0 not 1
    markers = markers+1
    
    #Mark the unknown region with 0
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(res, markers)
    res[markers == -1] = [255,0,0]
    placeholder = np.random.rand(224,224)
    #Displaying the markers on image
    markers = np.dstack([markers,np.zeros((224,224)), placeholder])
    #Adding the images into list
    preimgs.append(res)
    preimgs.append(markers)
    
    return preimgs

#-----------------*****************************************--------------------#

print('Preprocessing the model')

#Preprocessing and duplication of images
def preprocessing(img):
    height = 224
    width = 224
    pps_imgs = []
    dim = (width, height)
    res_img = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), dim, interpolation = cv2.INTER_LINEAR)
    res_img = res_img.astype(np.float32)/255
    pps_imgs.append(res_img)
    
    #Removing noise from image - Gaussian blur

    #blurred_img = cv2.GaussianBlur(res_img, (5,5),0)
    #pps_imgs.append(blurred_img)
    return pps_imgs[0]

#Preprocessing the images

def max_preprocessing(img):
    height = 224
    width = 224
    pps_imgs = []
    dim = (width, height)
    res_img = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), dim, interpolation = cv2.INTER_LINEAR)
    res_img = res_img.astype(np.float32)/255
    pps_imgs.append(res_img)
    
    #Removing noise from image - Gaussian blur

    blurred_img = cv2.GaussianBlur(res_img, (5,5),0)
    pps_imgs.append(blurred_img)
    return pps_imgs[0]

# In[9]:

print('Building the model')
#Working on the model
def build_model():
    model = Sequential()
    input_size  = Input(shape = (224,224,3), name  =  'Input_Image')

    #Layer 1 - Deapth Layer 1,2
    x = Conv2D(64,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer1' )(input_size)
    x = Conv2D(64,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer2' )(x)
    x = MaxPool2D((2,2), name = 'Maxpool1')(x)
    x = BatchNormalization(name = 'bn1')(x)

    #Layer 2 - Deapth layer 3,4
    x = Conv2D(128,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer3')(x)
    x = Conv2D(128,(3,3), activation = 'relu', padding = 'same', name = 'ConvLayer4')(x)
    x = MaxPool2D((2,2), name = 'Maxpoo12')(x)
    x = BatchNormalization(name = 'bn2')(x)
#     #Layer 3 - Deapth layer 3
#     x = Conv2D(30,(3,3), activation= 'relu',padding = 'same',  name = 'ConvLayer3')(x)
#     x = MaxPool2D((2,2), name = 'Maxpool3')(x)

    #Flatten the model

    x = Flatten(name = 'Flatten')(x)
    x = Dense(1024, activation = 'relu', name = 'FC1')(x)
    x = Dense(1024, activation = 'relu', name = 'FC2')(x)
    x = Dropout(0.7, name = 'Dropout2')(x)
    x = Dense(6, activation = 'softmax', name = 'Fc3')(x)
    
    model = Model(input = input_size , output = x)
    return model

#Building the model and summary

model = build_model() 
model.summary()



#Compiling the model
opt = Adam(lr = 0.0001, decay = 1e-5)
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath= 'bestmodel',save_best_only=True, save_weights_only=True)
model.compile(loss= 'binary_crossentropy', metrics= ['accuracy'], optimizer= opt)

