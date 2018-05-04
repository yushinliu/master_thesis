import numpy as np
import pandas as pd
import scipy
import cv2
import os
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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

def psnr(output_img,input_img,target_img,shrunk_size=50,save_dir="saved_models"):
    """
    calculate the bicubic of test_set
    """
    #targe_new_img=target-np.mean(target_img)
    bicubic_set=[]
    edsr_set=[]
    os.makedirs(save_dir+"//image//input")
    os.makedirs(save_dir+"//image//output")
    os.makedirs(save_dir+"//image//target")
    for index in range(input_img.shape[0]):
        input=input_img[index,:,:,:].reshape(input_img.shape[1],input_img.shape[1])
        cv2.imwrite(save_dir+"//image//input//"+str(index+2)+".png",input)
        output=output_img[index,:,:,:].reshape(output_img.shape[1],output_img.shape[1])
        cv2.imwrite(save_dir+"//image//output//"+str(index+2)+".png",output)
        target=target_img[index,:,:,:].reshape(target_img.shape[1],target_img.shape[1])
        cv2.imwrite(save_dir+"//image//target//"+str(index+2)+".png",target)
        #target_new=target_new_img[index,:,:,:].reshape(target_new_img.shape[1],target_new_img.shape[1])
        #bicubic_img=scipy.misc.imresize(input,(120,120),interp='bicubic')
        bicubic_img=cv2.resize(input,(100,100),interpolation=cv2.INTER_CUBIC)
        bicubic_set.append(new_psnr(bicubic_img,target))
        edsr_set.append(new_psnr(output,target))
    bicubic_mean=np.mean(bicubic_set)
    edsr_mean=np.mean(edsr_set)
    bicubic_set.append(bicubic_mean)
    edsr_set.append(edsr_mean)


    print("bicubic_set average is ",np.mean(bicubic_set))
    print("edsr_set average is ",np.mean(edsr_set))
    
    """
    save the bicubic result
    """
    psnr_set=pd.DataFrame({'bicubic':bicubic_set,'edsr':edsr_set})
    psnr_set.to_csv(save_dir+"//psnr_result.csv",index=False)
    return psnr_set

def dist_diagram(psnr_set,save_dir="saved_models"):
	new_table=psnr_set['edsr'] - psnr_set['bicubic']
	image=sns.distplot(new_table,rug=True)
	plt.xlabel("difference between edsr and bicubic")
	plt.ylabel("ratio")
	fig=image.get_figure()
	fig.savefig(save_dir+"//image//hist.png")
	fig.clf()

