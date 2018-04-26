# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:19:59 2018

@author: sunkg
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
Creates a convolutional residual block
as defined in the paper. More on
this inside model.py

x: input to pass through the residual block
channels: number of channels to compute
stride: convolution stride
"""
def resBlock(x,features=64,kernel_size=[3,3],scaling_factor=1, index=None):
	tmp = slim.conv2d(x,features,kernel_size,activation_fn=None, scope='conv%03d_1' % index)
	tmp = tf.nn.relu(tmp)
	tmp = slim.conv2d(tmp,features,kernel_size,activation_fn=None, scope='conv%03d_2' % index)
	tmp *= scaling_factor
	return x + tmp
 
def deconv2d(x, scale, features=64, kernel_size=[3,3], padding = "SAME"):
    img_height, img_width = int(x.shape[1]*scale), int(x.shape[2]*scale)
    filter_shape = [kernel_size[0], kernel_size[1], features, x.shape[3]]
    output_shape = [x.shape[0], img_height, img_width, features]
    strides = [1, 1, 1, 1] # be sure if 1,1,1,1 is correct or using interpolation
    return tf.nn.conv2d_transpose(x, filter_shape, output_shape, strides, padding)
    
def superResBlock(x, features=64, kernel_size=[3,3], scaling_factor = 0.1, repeat = 8, channels = 1):
    conv_origin = x
    #tmp = slim.conv2d(x,features,kernel_size,activation_fn=None, scope='conv%03d_1' % index)
    for i in range(repeat):
        x = resBlock(x,features,kernel_size, scaling_factor, repeat)
    #x = slim.conv2d(x, channels, kernel_size,activation_fn=None)
    x += conv_origin
    return x


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
