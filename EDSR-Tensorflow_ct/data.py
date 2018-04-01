import scipy.misc
import random
import numpy as np
import os

train_set = []
test_set = []
batch_index = 0
SEED=1

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images
"""
def load_dataset(data_dir):

	global train_set 
	global test_set 
	global name_set


	train_set = []
	test_set = []
	name_set = []
	SEED = 1

	"""
	extract the TRIAN data

	"""
	imgs = {}
	try:
		for items in os.listdir(str(data_dir)+"//DATASET-Train-augmented-120"):
			img_set = []
			name_set.append(items)
			count = 0
			for addr_1 in os.listdir(str(data_dir)+"//DATASET-Train-augmented-120//"+str(items)):
				for img in os.listdir(str(data_dir)+"//DATASET-Train-augmented-120//"+str(items)+"//"+str(addr_1)):
					#print("img",img)
					count +=1
					img_set.append(scipy.misc.imread(str(data_dir)+"//DATASET-Train-augmented-120//"+str(items)+"//"+str(addr_1)+"//"+str(img)))
			print(" the "+str(items)+" is "+str(count))
			random.seed(SEED)
			random.shuffle(img_set)
			imgs[items] = img_set[:100]
		print("train data extract finsihed")
	except:
		print("train wrong")

	for items in name_set:
		train_set.extend(imgs[items])
	random.shuffle(train_set)

	"""
	extract the TEST data
	"""
	imgs = {}
	try:
		for items in os.listdir(str(data_dir)+"//DATASET-Test-120"):
			img_set=[]
			count = 0
			for addr_1 in os.listdir(str(data_dir)+"//DATASET-Test-120//"+str(items)):
				for img in os.listdir(str(data_dir)+"//DATASET-Test-120//"+str(items)+"//"+str(addr_1)):
					#print("img",img)
					count +=1
					img_set.append(scipy.misc.imread(str(data_dir)+"//DATASET-Test-120//"+str(items)+"//"+str(addr_1)+"//"+str(img)))
			print(" the "+str(items)+" is "+str(count))
			random.seed(SEED)
			random.shuffle(img_set)
			imgs[items] = img_set[:20]
		print("test data extract finsihed")
	except:
		print("test wrong")
	'''
	for items in name_set:
		dataset = train_set[items]
		batch_size[items] = np.int(len(dataset)/batch_number)
	print(batch_size)
	'''
	for items in name_set:
		test_set.extend(imgs[items])
	random.shuffle(test_set)

	return train_set,test_set

"""
Get test set from the loaded dataset

size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.

returns the test set of your data
"""
def get_test_set(shrunk_size):
	"""for i in range(len(test_set)):
		img = scipy.misc.imread(test_set[i])
		if img.shape:
			img = crop_center(img,original_size,original_size)		
			x_img = scipy.misc.imresize(img,(shrunk_size,shrunk_size))
			y_imgs.append(img)
			x_imgs.append(x_img)"""

	x = [change_image(scipy.misc.imresize(q,(shrunk_size,shrunk_size))) for q in test_set]
	y = [change_image(q) for q in test_set]

	return x,y

def change_image(imgtuple):
	img = imgtuple[:,:,np.newaxis]
	return img
	

"""
Get a batch of images from the training
set of images.

batch_size: size of the batch
original_size: size for target images
shrunk_size: size for shrunk images

returns x,y where:
	-x is the input set of shape [-1,shrunk_size,shrunk_size,channels]
	-y is the target set of shape [-1,original_size,original_size,channels]
"""
def get_batch(batch_size,shrunk_size):
	global batch_index

	target_img = []
	input_img = []

	max_counter = len(train_set)/batch_size
	counter = batch_index % max_counter
	try:
		imgs = train_set[batch_size*int(counter):batch_size*(int(counter)+1)]
		x = [change_image(scipy.misc.imresize(q,(shrunk_size,shrunk_size))) for q in imgs]
		y = [change_image(q) for q in imgs] 
	except:
		print("wrong")

	batch_index = (batch_index+1) % max_counter
	return x,y,batch_index

"""
Simple method to crop center of image

img: image to crop
cropx: width of crop
cropy: height of crop
returns cropped image
"""
def crop_center(img,cropx,cropy):
	SEED=1
	random.seed(SEED)
	y,x,_ = img.shape
	#random.sample: any number between x-cropx-1 and 1
	startx = random.sample(range(x-cropx-1),1)[0]#x//2-(cropx//2)
	print(startx)
	starty = random.sample(range(y-cropy-1),1)[0]#y//2-(cropy//2)
	return img[starty:starty+cropy,startx:startx+cropx]



