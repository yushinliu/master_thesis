# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:04:46 2018

@author: sunkg
"""

import tensorflow.contrib.slim as slim
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import shutil
import utils
import os
import sys

"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class ResGUN(object):

    def __init__(self, img_size=32, num_layers=8, feature_size=64, scale=16, output_channels=1, image_depth = 16, iteration=1000):
        print("Building EDSR...")
        self.img_size = img_size
        self.output_channels = output_channels
        self.iteration = iteration
        self.image_depth = image_depth
        self.feature_size = feature_size
        self.num_layers = num_layers        
        
        self.input = tf.placeholder(tf.float32, [None, img_size, img_size, output_channels])
        self.target = tf.placeholder(tf.float32)
        self.scale = tf.placeholder(tf.float32)
        self.isIni = tf.placeholder(tf.bool)

        self.scaling_factor = 0.1
        self.scale_list = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]
        self.namespace_list = list()
        self.build_model()
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("PSNR", self.PSNR)
        #Image summaries for input, target, and output
        tf.summary.image("input_image",tf.cast(self.input, tf.uint8))
        tf.summary.image("target_image",tf.cast(self.target, tf.uint8))
        tf.summary.image("output_image",tf.cast(self.out, tf.uint8))

        
        
    def build_model(self):	
        """
	   Preprocessing as mentioned in the paper, by subtracting the mean
	   However, the subtract the mean of the entire dataset they use. As of
	   now, I am subtracting the mean of each batch
	   """   
        #self.scale = tf.placeholder(tf.float32)

        #Placeholder for image inputs
        #self.input = tf.placeholder(tf.float32,[None, self.img_size, self.img_size, self.output_channels])
        #self.input = tf.placeholder(tf.float32)
	   #Placeholder for ground-truth        
        #self.target = tf.placeholder(tf.float32,[None, self.img_size*self.scale, self.img_size*self.scale, self.output_channels]) 
        #self.target = tf.placeholder(tf.float32)
        
        #self.scaling_factor = 0.1
        #self.scale_list = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]
        self.scale = self.scale_list[-1]
        self.isIni = True
        mean_x = tf.reduce_mean(self.input)
        image_input =self.input - mean_x
        mean_y = tf.reduce_mean(self.target)
        image_target =self.target - mean_y
        '''
        for m in range (len(self.scale_list)):   
            with tf.name_scope("SuperBlock%2d" %m):
                x = slim.conv2d(image_input, self.feature_size,[3,3])   
                x = utils.superResBlock(x, self.feature_size, [3,3], self.scaling_factor, self.num_layers, self.output_channels)	
                x = utils.deconv2d(x, self.scale_list[m], self.output_channels, [3,3])
                if (self.scale_list[m] == self.scale):
                    x = utils.resBlock(x, self.feature_size, [3,3])
                    break
                elif (m == len(self.scale_list)-1):
                    sys.exit('Model structur does not fit specified scale.')
        '''
                    
	   #One final convolution on the upsampling output
        output = self.forward(image_input, image_target)
        self.out = tf.clip_by_value(output+mean_x, 0.0, np.power(2, self.image_depth)-1)

        self.loss = tf.reduce_mean(tf.losses.absolute_difference(image_target,output))
	
        print("Done building!")
        
        
    def forward(self, image_input, image_target):
        
        if self.isIni:
            with tf.name_scope("StartConv2d"):
                self.namespace_list.append(tf.get_default_graph().get_name_scope())
                with tf.variable_scope("VariableConv2d"): 
                    x = slim.conv2d(image_input, self.feature_size,[3,3])  
                    
            for m in range (len(self.scale_list)):  
         
                with tf.name_scope("SuperBlock"): 
                    self.namespace_list.append(tf.get_default_graph().get_name_scope())
                    with tf.variable_scope("VariableSB"):
                        x = utils.superResBlock(x, self.feature_size, [3,3], self.scaling_factor, self.num_layers, self.output_channels)	
                if (self.scale_list[m] == self.scale):
                    x = slim.conv2d(image_input, self.output_channels, [3,3]) 
                    break 
                else:
                    with tf.name_scope("Deconv2d"):
                        self.namespace_list.append(tf.get_default_graph().get_name_scope())
                        with tf.variable_scope("VariableDeconv2d"):
                            x = utils.deconv2d(x, self.scale_list[m+1], self.output_channels, [3,3])
        
        elif not self.isIni:
            with tf.variable_scope(self.namespace_list[0] + "/VariableConv2d", reuse = True): 
                    x = slim.conv2d(image_input, self.feature_size,[3,3])  
                    
            for m in range (len(self.scale_list)):  
         
                with tf.variable_scope(self.namespace_list[2*m+1] + "/VariableSB", reuse = True):
                    x = utils.superResBlock(x, self.feature_size, [3,3], self.scaling_factor, self.num_layers, self.output_channels)	
                if (self.scale_list[m] == self.scale):
                    x = slim.conv2d(image_input, self.output_channels, [3,3]) 
                    break 
                else:
                    with tf.variable_scope(self.namespace_list[2*m+2] + "/VariableDeconv2d", reuse = True):
                        x = utils.deconv2d(x, self.scale_list[m+1], self.output_channels, [3,3])
                                    
        return x
        
	
    """
    Save the current state of the network to file
    """
    def save(self,savedir='saved_models'):
        print("Saving...")
        self.saver.save(self.sess,savedir+"/model")
        print("Saved!")
		
    """
    Resume network from previously saved weights
    """
    def resume(self,savedir='saved_models'):
        print("Restoring...")
        self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))
        print("Restored!")	

    """
    Compute the output of this network given a specific input

    x: either one of these things:
		1. A numpy array of shape [image_width,image_height,3]
		2. A numpy array of shape [n,input_size,input_size,3]

    return: 	For the first case, we go over the entire image and run super-resolution over windows of the image
			that are of size [input_size,input_size,3]. We then stitch the output of these back together into the
			new super-resolution image and return that

    return  	For the second case, we return a numpy array of shape [n,input_size*scale,input_size*scale,3]
    """
    def predict(self,x, scale):
        print("Predicting...")
        if (len(x.shape) == 3) and not(x.shape[0] == self.img_size and x.shape[1] == self.img_size):
            num_across = x.shape[0]//self.img_size
            num_down = x.shape[1]//self.img_size
            tmp_image = np.zeros([x.shape[0]*self.scale,x.shape[1]*self.scale,self.output_channels])
            for i in range(num_across):
                for j in range(num_down):
                    tmp = self.sess.run(self.out,feed_dict={self.input:[x[i*self.img_size:(i+1)*self.img_size,j*self.img_size:(j+1)*self.img_size]]})[0]
                    tmp_image[i*tmp.shape[0]:(i+1)*tmp.shape[0],j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
                #this added section fixes bottom right corner when testing
            if (x.shape[0]%self.img_size != 0 and  x.shape[1]%self.img_size != 0):
                 tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,-1*self.img_size:]]})[0]
                 tmp_image[-1*tmp.shape[0]:,-1*tmp.shape[1]:] = tmp
					
            if x.shape[0]%self.img_size != 0:
                 for j in range(num_down):
                     tmp = self.sess.run(self.out,feed_dict={self.input:[x[-1*self.img_size:,j*self.img_size:(j+1)*self.img_size]]})[0]
                     tmp_image[-1*tmp.shape[0]:,j*tmp.shape[1]:(j+1)*tmp.shape[1]] = tmp
            if x.shape[1]%self.img_size != 0:
                 for j in range(num_across):
                     tmp = self.sess.run(self.out,feed_dict={self.input:[x[j*self.img_size:(j+1)*self.img_size,-1*self.img_size:]]})[0]
                     tmp_image[j*tmp.shape[0]:(j+1)*tmp.shape[0],-1*tmp.shape[1]:] = tmp
            return tmp_image
        else:
            if (self.scale in self.scale_list):
                return self.sess.run(self.out,feed_dict={self.input:x, self.scale:scale})
            else:
                sys.exit('No fitting scale factor!')

    """
    Function to setup your input data pipeline
    """
    def set_data_fn(self,fn,args):
        self.data = fn
        self.args = args

    def cal_PSNR(self):
        mse = tf.reduce_mean(tf.squared_difference(self.target, self.out))	
        PSNR = tf.constant((np.power(2, self.image_depth)-1)**2,dtype=tf.float32)/mse
        self.PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)

    """
    Train the neural network
    """
    def train(self,save_dir="saved_models"):
        #Removing previous save directory if there is one
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
	   #Make new save directory
        os.mkdir(save_dir)

        merged = tf.summary.merge_all()
        #Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer()
        #This is the train operation for our objective
        train_op = optimizer.minimize(self.loss)	
        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            #create summary writer for train
            train_writer = tf.summary.FileWriter(save_dir+"/train",sess.graph)

            #This is our training loop
            for i in tqdm(range(self.iteration)):
                
            #Use the data function we were passed to get a batch every iteration
                x,y = self.data(*self.args)

				#Create feed dictionary for the batch
                feed = {
                       self.input:x,
                       self.target:y,
                       self.scale:y.shape[1]//x.shape[1],
                       self.isIni:True
				  }
			#Run the train op and calculate the train summary
                summary, _ , loss_val = sess.run([merged, train_op, self.loss],feed)
                
                if i %10 == 0:
                    #Write train summary for this step
                    train_writer.add_summary(summary,i)
                    print('Loss value at interation %s is %03d' %(i, loss_val))                
		 #Save our trained model		
            self.save()		

    def test(self,save_dir="saved_models"):

        merged = tf.summary.merge_all()
        
        #Scalar to keep track for loss

        print("Begin testing...")
        with self.sess as sess:
            #create summary writer for test in batch
            test_writer = tf.summary.FileWriter(save_dir+"/test")

            #This is our training loop
            for i in tqdm(range(self.iteration)):
            #Use the data function we were passed to get a batch every iteration
                x,y = self.data(*self.args)
			#Create feed dictionary for the batch
                feed = {
                        self.input:x, 
                        self.target:y,
                        self.scale:y.shape[1]//x.shape[1],
                        self.isIni:False
                        }
                summary, PSNR= sess.run([merged, self.PSNR], feed)
                #Calculating Peak Signal-to-noise-ratio
                #Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
                if i %10 == 0:                   
                    #Write train summary for this step
                    test_writer.add_summary(summary,i)
                    print('PSNR at interation %s is %03d' %(i, PSNR))

