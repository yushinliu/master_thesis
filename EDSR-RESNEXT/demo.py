import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import resxBlock


x = tf.placeholder(tf.float32,[None,60,60,1])

