import numpy as np
#import pickle
import re
import random
import os
#from PIL import Image
#from cv2 import imread,imwrite
import shutil


def extract_data(list_dir):
    img={}
    try:
        for addr in os.listdir(list_dir):
            f = open(list_dir+"//"+str(addr),'r')
            train_id = []
            for line in f.readlines():
                tmp=line.split("\\")[4]
                train_id.append(tmp.split("	")[0])
            f.close()
            random.shuffle(train_id)
            train_set=train_id[:np.int(len(train_id)*0.05)]
            img[addr]=train_set
    except:
        print("errors occur")
    return img


def re_allocate(img,img_dir,save_dir):
    os.makedir(save_dir+"//train_color")
    os.makedir(save_dir+"//train_label")
    try:
        for id in img.keys():
            for img_name in img[id]:
                shutil.copy(img_dir+"//"+"train_color"+"//"+img_name,save_dir+"//train_color")
                label_name=img_name[:-4]+"_instanceIds.png"
                shutil.copy(img_dir+"//"+"train_label"+"//"+label_name,save_dir+"//label_color")
    except:
        print("errors occur")


"""
generate data start
"""
img=extract_data(list_dir)
re_allocate(img,img_dir,save_dir)
