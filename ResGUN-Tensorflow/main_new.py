from utils.data import *
from utils.psnr_cal import *
from model.ResGun_2x import ResGUN
import os

'''
load_dataset("/home/liuyn/masterthesis/master_thesis/dataset",500) #arg: (dir,batch_number)
network=EDSR(60,16,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(500,60),get_test_set,60)
network.train()
'''

os.environ['CUDA_VISIBLE_DEVICES'] ='7'
#load_dataset("D://LAB//master_thesis//dataset//General-100",100) #arg: (dir,batch_number)
load_dataset("/home/liuyn/masterthesis/master_thesis/dataset/General-100",100) #arg: (dir,img_size)
network=ResGUN(50,4,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(16,50),get_test_set,(50))
input_img,target_img,output_img=network.train(300,0.95,6000,save_dir="saved_models_general100_origin_2")
psnr_set=psnr(output_img,input_img,target_img,save_dir="saved_models_general100_origin_2")
dist_diagram(psnr_set,save_dir="saved_models_general100_origin_2")

