from utils.data import *
from utils.psnr_cal import *
from model.ResGun_2x import ResGUN
import cv2
import os

'''
load_dataset("/home/liuyn/masterthesis/master_thesis/dataset",500) #arg: (dir,batch_number)
network=EDSR(60,16,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(500,60),get_test_set,60)
network.train()
'''
class ResGUN_predict(ResGUN):
      def predict_image(self,input_image,target_image,SRCNN_image,bicubic_image,model_dir=''):
          SEED = 1
          self.resume(savedir=model_dir)
          x,y,z=input_image.shape
          coords_x = x//self.img_size
          coords_y = y//self.img_size
          coords = [ (q,r) for q in range(int(coords_x)) for r in range(int(coords_y)) ] 
          random.seed(SEED)
          random.shuffle(coords)
          input_img=input_image[coords[0][0]*self.img_size:(coords[0][0]+1)*self.img_size,coords[0][1]*self.img_size:(coords[0][1]+1)*self.img_size,:]
          input_img=input_img.reshape((1,)+input_img.shape)
          output_img=self.sess.run(self.out,feed_dict={self.input:input_img})

          target_img=target_image[coords[0][0]*self.img_size*self.scale:(coords[0][0]+1)*self.img_size*self.scale,coords[0][1]*self.img_size*self.scale:(coords[0][1]+1)*self.img_size*self.scale,:]
          bicubic_img=bicubic_image[coords[0][0]*self.img_size*self.scale:(coords[0][0]+1)*self.img_size*self.scale,coords[0][1]*self.img_size*self.scale:(coords[0][1]+1)*self.img_size*self.scale,:]
          SRCNN_img=SRCNN_image[coords[0][0]*self.img_size*self.scale:(coords[0][0]+1)*self.img_size*self.scale,coords[0][1]*self.img_size*self.scale:(coords[0][1]+1)*self.img_size*self.scale,:]
          return output_img,target_img,bicubic_img,SRCNN_img
if __name__ == '__main__': 
  os.environ['CUDA_VISIBLE_DEVICES'] ='3'
  #load_dataset("D://LAB//master_thesis//dataset//General-100",100) #arg: (dir,batch_number)
  load_dataset("/home/liuyn/masterthesis/master_thesis/dataset/General-100",100) #arg: (dir,img_size)
  network=ResGUN_predict(50,8,64,2) #ONE BASELINE
  input_image=cv2.imread('/home/liuyn/masterthesis/master_thesis/dataset/Set14_SR/Set14/image_SRF_2/img_005_SRF_2_LR.png')
  target_image=cv2.imread('/home/liuyn/masterthesis/master_thesis/dataset/Set14_SR/Set14/image_SRF_2/img_005_SRF_2_HR.png')
  bicubic_image=cv2.imread('/home/liuyn/masterthesis/master_thesis/dataset/Set14_SR/Set14/image_SRF_2/img_005_SRF_2_bicubic.png')
  SRCNN_image=cv2.imread('/home/liuyn/masterthesis/master_thesis/dataset/Set14_SR/Set14/image_SRF_2/img_005_SRF_2_SRCNN.png')
  output_img,target_img,bicubic_img,SRCNN_img=network.predict_image(input_image,target_image,SRCNN_image,bicubic_image,model_dir='/home/liuyn/masterthesis/mt_result/result/5_13/saved_models_general100_bicubic')
  print("the output psnr is ",new_psnr(output_img,target_img))
  print("the bicubic psnr is ",new_psnr(bicubic_img,target_img))
  print("the SRCNN psnr is ",new_psnr(SRCNN_img,target_img))
  #psnr_set=psnr(output_img,input_img,target_img,save_dir="saved_models_general100_upsample_4x_all")
  #dist_diagram(psnr_set,save_dir="saved_models_general100_upsample_4x_all")

