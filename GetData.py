#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time



#Reading the dataset

print('Reading the data')
#Reading the data
member = ("C:\\Users\\BlueFlames\\Python Projects\\Datasets\\Garbage classification\\Garbage classification\\")
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

#Splitting the dataset into training, test and validation set

print('Splitting the dataset')
train_data = data[1:2000]
val_data = data[2001:2200]
test_data = data[2201:2527]

