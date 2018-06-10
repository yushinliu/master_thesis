
import scipy.misc
import tensorflow as tf
import numpy as np
import shutil
import os
import time 
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] ='1,2'
SEED = 42
np.random.seed(SEED)
target_data=np.append(np.ones((1,8)),np.zeros((1,8)))
target_data=target_data.reshape(target_data.shape+(1,))
input_data=np.random.rand(1,16)


def test():
    weights_1= tf.get_variable('w1',[8,32])
    bias_1 =tf.get_variable('b1',[32])
    weights_2= tf.get_variable('w2',[32,8])
    bias_2 =tf.get_variable('b2',[8])
    opt = tf.train.GradientDescentOptimizer(0.001)
    tower_grads=[]
    for i in range(2):
        with tf.device('/gpu:%d' %i ):
             with tf.name_scope('cv%d' %i):
                  x = tf.placeholder(tf.float32,[1,8],name='x')
                  y = tf.placeholder(tf.float32,[8,1],name='y')
                  a1=tf.matmul(x,weights_1)+bias_1
                  o1=tf.sigmoid(a1)
                  a2=tf.matmul(o1,weights_2)+bias_2
                  o2=tf.sigmoid(a2)
                  loss = tf.reduce_mean(tf.squared_difference(y,o2))
                  grads=opt.compute_gradients(loss)
                  #grad=tf.Print(grads,grads)
                  tower_grads.append(grads)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    with sess as sess:
         #writer=tf.summary.FileWriter("test_graph",sess.graph)
         sess.run(init)
         #print(y.name)
         #print("y is ")
         #sess.run(grads,feed_dict={'cv0/x:0':input_data[:,:8],'cv1/x:0':input_data[:,8:],'cv0/y:0':target_data[:8,:],'cv1/y:0':target_data[8:,:]}) #runing at the same time
         grd1,grd2=sess.run([tower_grads[0],tower_grads[1]],feed_dict={'cv0/x:0':input_data[:,:8],'cv1/x:0':input_data[:,8:],'cv0/y:0':target_data[:8,:],'cv1/y:0':target_data[8:,:]}) #runing at the same time
         print(grd1)
         print("------------------------------------------------------------")
         print(grd2)
         print(len(tower_grads))
         #tf.train.Saver.save(sess,"./model")
         #writer.close()

test()
