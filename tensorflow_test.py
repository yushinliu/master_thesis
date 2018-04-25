
import scipy.misc
import tensorflow as tf
import numpy as np
import shutil
import os
import time 
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] ='3,4,5'
SEED = 42
def cv_data(SEED):
	np.random.seed(SEED)
	return np.random.rand(5,2,2)


def test(data):
	for i in range(3):
		with tf.device('/gpu:%d' %i ):
			with tf.name_scope('cv%d' %i):
				x = tf.placeholder(tf.float32,[2,2],name='x')
				y = tf.matmul(x,x)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	with sess as sess:
		writer=tf.summary.FileWriter("test_graph",sess.graph)
		sess.run(init)
		#print(y.name)
		#print("y is ")
		while(True):
			sess.run(['cv2/MatMul:0','cv1/MatMul:0'],feed_dict={'cv0/x:0':np.ones((2,2)),'cv1/x:0':2*np.ones((2,2)),'cv2/x:0':3*np.ones((2,2))}) #runing at the same time
		#tf.train.Saver.save(sess,"./model")
		writer.close()

def main(SEED):
	dataset=cv_data(SEED)
	test(dataset)


main(SEED)
