import numpy as np
import pandas as pd
import scipy.misc
import cv2
import os

for dataset in ['battery','Biostone','PCB']:
	input_dir="result//4_3//"+dataset+"//image//input//"
	img_set=[]
	name_set=[]
	output_dir="result//4_3//"+dataset+"//image//bicubic//"
	os.makedirs(output_dir)
	for img in os.listdir(input_dir):
		name_set.append(img)
		img_temp=scipy.misc.imread(input_dir+str(img))
		img_set.append(img_temp)
	for index in range(len(img_set)):
		input=img_set[index]
		bicubic_img=cv2.resize(input,(120,120),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(output_dir+name_set[index],bicubic_img)
	
