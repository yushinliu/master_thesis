import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import scipy.misc
import cv2
import os
train_dir="/home/liuyn/masterthesis/master_thesis/dataset/DATASET-Train-augmented-120"
test_dir="/home/liuyn/masterthesis/master_thesis/dataset/DATASET-Test-120"

def new_psnr(input,output):
    
    mse=np.mean((input-output)**2)
    psnr = (255**2)/mse
    psnr = 10 * np.log10(psnr)
    
    """
    x = tf.placeholder(tf.float32,[120,120])
    y = tf.placeholder(tf.float32,[120,120])
    mse = tf.reduce_mean(tf.squared_difference(x,y))
    PSNR = tf.constant(255**2,dtype=tf.float32)/mse
    PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)
    
    with tf.Session() as sess:
        psnr=sess.run(PSNR,feed_dict={x:input,y:output})
    """
    return psnr




for data_dir in [train_dir]:
    dataset={}
    for item in ['battery','BioStone','PCB']:
        img_set=[]
        for files in os.listdir(data_dir+"//"+str(item)):
            for img in os.listdir(data_dir+"//"+str(item)+"//"+str(files)):
                image=scipy.misc.imread(data_dir+"//"+str(item)+"//"+str(files)+"//"+str(img))
                img_set.append(image)
        dataset[item]=img_set
    if data_dir == train_dir:
        print("train extraction compeleted")
    else :
        print("test extraction compeleted")

    save_dir=data_dir+'//new_set_3//'
        
    for name in dataset.keys():
        temp_set=[0]
        os.makedirs(save_dir+str(name))
        if name == 'battery':
           for index,img in enumerate(dataset[name]):
               mark = 0
               if np.var(img) > 40:
                   for new_index in temp_set[::-1]:
                       new_img=dataset[name][new_index]
                       psnr=new_psnr(new_img,img)
                       if psnr > 29.8:
                           mark = 1
                           break
                   if mark == 1:
                       mark = 0
                       continue
                   cv2.imwrite(save_dir+str(name)+"//"+str(index)+".png",img)
                   temp_set.append(index)
        elif name == 'BioStone':
            for index,img in enumerate(dataset[name]):
                mark = 0
                if np.var(img) > 100:
                   for new_index in temp_set[::-1]:
                       new_img=dataset[name][new_index]
                       psnr=new_psnr(new_img,img)
                       if psnr > 28.5:
                           mark = 1
                           break
                   if mark == 1:
                       mark = 0
                       continue
                   cv2.imwrite(save_dir+str(name)+"//"+str(index)+".png",img)
                   temp_set.append(index)
        elif name == 'PCB':
            for index,img in enumerate(dataset[name]):
                mark = 0
                if np.var(img) > 25:
                   for new_index in temp_set[::-1]:
                       new_img=dataset[name][new_index]
                       psnr=new_psnr(new_img,img)
                       if psnr > 28.8:
                           mark = 1
                           break
                   if mark == 1:
                       mark = 0
                       continue
                   cv2.imwrite(save_dir+str(name)+"//"+str(index)+".png",img)
                   temp_set.append(index)
