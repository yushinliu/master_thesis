# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 08:41:50 2018

@author: sunkg
"""

from ResGUN_model import ResGUN
import scipy.misc
import argparse
import data
import os
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--image")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")

args = parser.parse_args()
#data.load_dataset(args.dataset,args.imgsize)
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)
    
down_size = args.imgsize//args.scale
network = ResGUN(down_size,args.layers,args.featuresize,scale=args.scale, output_channels=3, image_depth = 16, args.iterations)
network.resume(args.savedir)

if args.dataset:
    data.load_dataset(args.dataset,args.imgsize)
    network.set_data_fn(data.get_batch,('test', args.batchsize,args.imgsize,down_size))
    network.test(args.iterations,args.savedir)
elif args.image:
    x = scipy.misc.imread(args.image)
    inputs = x
    outputs = network.predict(x, args.scale)
    scipy.misc.imsave(args.outdir+"/input_"+args.image,inputs)
    scipy.misc.imsave(args.outdir+"/output_"+args.image,outputs)
else:
	print("No image argument given")
	

