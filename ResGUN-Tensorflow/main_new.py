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
if __name__ == "__main__":
   resume_dir = "/home/liuyn/masterthesis/master_thesis/ResGUN-Tensorflow/saved_models_general100_augmentated_newepoch_batch_size_64_1"
   os.environ['CUDA_VISIBLE_DEVICES'] ='1'
   #load_dataset("D://LAB//master_thesis//dataset//General-100",100) #arg: (dir,batch_number)
   load_dataset("/home/liuyn/masterthesis/master_thesis/dataset/Training",40) #arg: (dir,img_size)
   network=ResGUN(20,4,64,2) #ONE BASELINE
   network.set_data_fn(get_batch,(64,20),get_test_set,(20))
   input_img,target_img,output_img=network.train(1800,0.9,18000,save_dir="saved_models_general100_augmentated_newepoch_batch_size_64_2",resume=True,resume_dir=resume_dir)
   psnr_set=psnr(output_img,input_img,target_img,save_dir="saved_models_general100_augmentated_newepoch_batch_size_64_2")
   dist_diagram(psnr_set,save_dir="saved_models_general100_augmentated_newepoch_batch_size_64_2")

