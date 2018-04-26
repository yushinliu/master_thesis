# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:50:15 2018

@author: sunkg
"""

import cv2
import random
import os
import numpy as np

train_set = []
test_set = []
batch_index = 0

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images
"""
def load_dataset(data_dir, img_size):

    global train_set
    global test_set
    img_files = os.listdir(data_dir)
    test_size = int(len(img_files)*0.2)
    test_indices = random.sample(range(len(img_files)),test_size)
    for i in range(len(img_files)):
        #img = scipy.misc.imread(data_dir+img_files[i])
        if i in test_indices:
            test_set.append(data_dir+"/"+img_files[i])
        else:
            train_set.append(data_dir+"/"+img_files[i])
    return

def get_image_cropped(im_gt, scale):
    width = int(im_gt.shape[1]//scale*scale)
    height = int(im_gt.shape[0]//scale*scale)
    h_start = int((im_gt.shape[0]-height)/2)
    w_start = int((im_gt.shape[1]-width)/2)
    return im_gt[h_start:h_start+height, w_start:w_start+width]

    
    
def get_prepared_set(imgs, original_size, scale, crop = True): 

    y = [get_image_cropped(q, scale) for q in imgs]
    x = [cv2.resize(p, (p.shape[1]/scale, p.shape[0]/scale), 0, 0, cv2.INTER_CUBIC) for p in y]
    return x,y


def get_batch(flag, batch_size, original_size):
    
    global batch_index
    
    scale_list = [1.5, 2, 3, 4, 8, 16]
    batch_scale = np.random.choice(scale_list)
    if (flag == 'Train'):
        max_counter_train = len(train_set)/batch_size
        counter_train = batch_index % max_counter_train
        window = [x for x in range(counter_train*batch_size,(counter_train+1)*batch_size)]
        imgs = [train_set[q] for q in window]
        x,y = get_prepared_set(imgs, original_size, batch_scale)
        batch_index = (batch_index+1)%max_counter_train
    elif(flag == 'Test'):
        max_counter_test = len(test_set)/batch_size
        counter_test = batch_index % max_counter_test
        window = [x for x in range(counter_test*batch_size,(counter_test+1)*batch_size)]
        imgs = [test_set[q] for q in window]
        x,y = get_prepared_set(imgs, original_size, batch_scale)
        batch_index = (batch_index+1)%max_counter_test
    return x,y






