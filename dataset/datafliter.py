import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import scipy.misc
import cv2
import os
from tqdm import tqdm

train_dir="/home/liuyn/masterthesis/master_thesis/dataset/DATASET-Train-augmented-120"
test_dir="/home/liuyn/masterthesis/master_thesis/dataset/DATASET-Test-120"


for data_dir in [train_dir,test_dir]:
    dataset={}
    for item in os.listdir(data_dir):
        img_set=[]
        for files in os.listdir(data_dir+"//"+str(item)):
            for img in os.listdir(data_dir+"//"+str(item)+"//"+str(files)):
                image=scipy.misc.imread(data_dir+"//"+str(item)+"//"+str(files)+"//"+str(img))
                img_set.append(image)
        dataset[item]=img_set

    save_dir=data_dir+'//new_set//'
        
    for name in dataset.keys():
        os.makedirs(save_dir+str(name))
        if name == 'battery':
           for index,img in enumerate(dataset[name]):
               if np.var(img) > 20:
                   cv2.imwrite(save_dir+str(name)+"//"+str(index)+".png",img)
        elif name == 'BioStone':
            for index,img in enumerate(dataset[name]):
                if np.var(img) > 80:
                    cv2.imwrite(save_dir+str(name)+"//"+str(index)+".png",img)
        elif name == 'PCB':
            for index,img in enumerate(dataset[name]):
                if np.var(img) > 3 and np.mean(img)>35:
                    cv2.imwrite(save_dir+str(name)+"//"+str(index)+".png",img)
