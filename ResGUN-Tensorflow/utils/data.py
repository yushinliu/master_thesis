# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:50:15 2018

@author: sunkg
"""

import cv2
import random
import os
import numpy as np
import scipy
import scipy.misc
from skimage import exposure

train_set = []
test_set = []
batch_index = 0

"""
data augmentation
"""
def rotation(image, random_flip=True):  
    if random_flip : #and np.random.choice([True, False]):  
        w,h = image.shape[1], image.shape[0]  
        # 0-360 random generate rotation 
        angle = np.random.randint(0,360)  
        RotateMatrix = cv2.getRotationMatrix2D(center=(image.shape[1]/2, image.shape[0]/2), angle=angle, scale=0.7)  
        # image = cv2.warpAffine(image, RotateMatrix, (w,h), borderValue=(129,137,130))  
        #image = cv2.warpAffine(image, RotateMatrix, (w,h),borderValue=(129,137,130))  
        image = cv2.warpAffine(image, RotateMatrix, (w,h),borderMode=cv2.BORDER_REPLICATE)  
    return image  
  
def gen_exposure(image, random_xp=True):  
    if random_xp and np.random.choice([True, False]):  
        image = exposure.adjust_gamma(image, 1.2) # darker
    if random_xp and np.random.choice([True, False]):  
        image = exposure.adjust_gamma(image, 1.5) #  darker
    if random_xp and np.random.choice([True, False]):  
        image = exposure.adjust_gamma(image, 0.9) # brighter  
    if random_xp and np.random.choice([True, False]):  
        image = exposure.adjust_gamma(image, 0.8) # brighter
    if random_xp and np.random.choice([True, False]):  
        image = exposure.adjust_gamma(image, 0.7) # darker 
    return image  

def data_augmentation(train_image,multi_num=3):
    temp = train_image.copy()
    print("data augmentation start")
    for img in temp:
        for i in range(multi_num):
            #print(img.shape)
            image=rotation(img)
            image=gen_exposure(image)
            train_image.append(image)
    print("data augmentation end")
    return train_image
    
"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images -- nature image
"""

SEED = 1

def load_dataset(data_dir, img_size):

    global train_set
    global test_set
    imgs = []
    img_files = os.listdir(data_dir)
    for img in img_files:
        try:
            #print("good")
            tmp= cv2.imread(data_dir+"//"+img) #read each image
            x,y,z = tmp.shape
            coords_x = x / img_size
            coords_y = y/img_size
            coords = [ (q,r) for q in range(int(coords_x)) for r in range(int(coords_y)) ]
            for coord in coords:
                imgs.append(tmp[coord[0]*img_size:(coord[0]+1)*img_size,coord[1]*img_size:(coord[1]+1)*img_size,:])
        except:
            print ("oops")
    #print(len(imgs))
    test_size = min(100,int( len(imgs)*0.1))
    #print(test_size)
    random.seed(SEED)
    random.shuffle(imgs)
    test_set = imgs[:test_size]
    train_set = imgs[test_size:]
    train_set = data_augmentation(train_set,multi_num=10)
    random.shuffle(train_set)
    print(len(train_set))
    return train_set,test_set


"""
Get test set from the loaded dataset

size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.

returns the test set of your data
"""
def get_test_set(shrunk_size):

    x = [scipy.misc.imresize(q,(shrunk_size,shrunk_size)) for q in test_set]
    y = [q for q in test_set]

    return x,y

#change_image only for ct image
def change_image(imgtuple):
    img = imgtuple[:,:,np.newaxis]
    return img
    

"""
Get a batch of images from the training
set of images.

batch_size: size of the batch
original_size: size for target images
shrunk_size: size for shrunk images

returns x,y where:
    -x is the input set of shape [-1,shrunk_size,shrunk_size,channels]
    -y is the target set of shape [-1,original_size,original_size,channels]
"""
def get_batch(batch_size,shrunk_size):
    global batch_index

    target_img = []
    input_img = []
    #print(len(train_set))

    max_counter = len(train_set)/batch_size
    counter = batch_index % max_counter

    imgs = train_set[batch_size*int(counter):batch_size*(int(counter)+1)]
    x = [scipy.misc.imresize(q,(shrunk_size,shrunk_size)) for q in imgs]
    y = [q for q in imgs] 

    batch_index = (batch_index+1) % max_counter
    return x,y,batch_index  #x,y is the list of batch image

"""
Simple method to crop center of image

img: image to crop
cropx: width of crop
cropy: height of crop
returns cropped image
"""






