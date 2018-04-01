from model_lr_decay import EDSR
from data import load_dataset,crop_center,get_batch,get_test_set
from psnr_cal import psnr,new_psnr
import os

'''
load_dataset("/home/liuyn/masterthesis/master_thesis/dataset",500) #arg: (dir,batch_number)
network=EDSR(60,16,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(500,60),get_test_set,60)
network.train()
'''
os.environ['CUDA_VISIBLE_DEVICES'] ='5,6,7'

load_dataset("/home/liuyn/masterthesis/master_thesis/dataset") #arg: (dir,batch_number)
network=EDSR(60,16,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(10,60),get_test_set,60)
input_img,target_img,output_img=network.train(300,0.9,3000)
psnr(output_img,input_img,target_img)

