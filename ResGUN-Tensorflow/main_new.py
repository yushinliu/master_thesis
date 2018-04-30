from utils import *
from model import ResGun_2x
import os

'''
load_dataset("/home/liuyn/masterthesis/master_thesis/dataset",500) #arg: (dir,batch_number)
network=EDSR(60,16,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(500,60),get_test_set,60)
network.train()
'''

os.environ['CUDA_VISIBLE_DEVICES'] ='3,4,5'
load_dataset("/home/liuyn/masterthesis/master_thesis/dataset",100) #arg: (dir,batch_number)
network=ResGun_2x(50,16,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(10,50),get_test_set,(50))
input_img,target_img,output_img=network.train(300,0.9,5000,save_dir="saved_models_general100")
psnr_set=psnr(output_img,input_img,target_img,save_dir="saved_models_general100")
dist_diagram(psnr_set,save_dir="saved_models_general100")

