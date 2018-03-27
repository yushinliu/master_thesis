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
def load_dataset(data_dir,batch_number):

	global train_set 
	global test_set 
	global name_set
	global batch_size

	train_set = {}
	test_set = {}
	batch_size = {}
	name_set = []

	"""
	extract the TRIAN data

	"""
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
			train_set[items] = img_set
		print("train data extract finsihed")
	except:
		print("train wrong")

	"""
	extract the TEST data
	"""

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
			test_set[items] = img_set
		print("test data extract finsihed")
	except:
		print("test wrong")

	for items in name_set:
		dataset = train_set[items]
		batch_size[items] = np.int(len(dataset)/batch_number)
	print(batch_size)

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
	img = []
	for items in name_set:
		img.extend(test_set[items])
	random.seed(SEED)
	random.shuffle(img)
	print(len(img))
	x = [change_image(scipy.misc.imresize(q,(shrunk_size,shrunk_size))) for q in img]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size].resize(shrunk_size,shrunk_size) for q in imgs]
	y = [change_image(q) for q in img]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size] for q in imgs]
	return x[:800],y[:800]

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
def get_batch(batch_number,shrunk_size):
	global batch_index

	target_img = []
	input_img = []

	
	counter = batch_index % batch_number
	try:
		if counter < (batch_number-1):
			for items in name_set:
				imgs = train_set[items][batch_size[items]*batch_index:batch_size[items]*(batch_index+1)]
				x = [change_image(scipy.misc.imresize(q,(shrunk_size,shrunk_size))) for q in imgs]
				y = [change_image(q) for q in imgs] 
			input_img.extend(x)
			target_img.extend(y)
		elif counter == batch_number-1:
			for items in name_set:
				imgs = train_set[items][batch_size[items]*batch_index:batch_size[items]*(batch_index+1)]
				x = [change_image(scipy.misc.imresize(q,(shrunk_size,shrunk_size))) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size].resize(shrunk_size,shrunk_size) for q in imgs]
				y = [change_image(q) for q in imgs]#scipy.misc.imread(q[0])[q[1][0]*original_size:(q[1][0]+1)*original_size,q[1][1]*original_size:(q[1][1]+1)*original_size] for q in imgs]
			input_img.extend(x)
			target_img.extend(y)
	except:
		if counter <= batch_number -1:
			print("wrong")
		else:
			print("finished")


	batch_index = (batch_index+1)% batch_number
	return input_img,target_img,batch_index

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





