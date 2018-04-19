import tensorflow.contrib.slim as slim
import scipy.misc
import tensorflow as tf
#from tqdm import tqdm
from process_bar import ShowProcess
import numpy as np
import shutil
import utils
import os
import time 
from sklearn.model_selection import KFold

SEED = 42
def cv_data(SEED):
	np.random.seed(SEED)
	return np.random.rand(5,2,2)


def test(data):
	for i in range(5):
		with tf.device('/gpu:%d' %i ):
			with tf.name_scope('cv%d' %i):
				x = tf.placeholder(tf.float32,[2,2])
				y = tf.matmul(x,x)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	with sess as sess:
		writer=tf.summary.FileWriter("test_graph",sess.graph)
		sess.run(init)
		print("y is ")
		print(sess.run(y,feed_dict={'cv0/x:0':data[0],'cv1/x:0':data[1],'cv2/x:0':data[2],'cv3/x:0':data[3],'cv4/x:0':data[4]}))
		writer.close()

def main(SEED):
	dataset=cv_data(SEED)
	test(dataset)


main(SEED)
