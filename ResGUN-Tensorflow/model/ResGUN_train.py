# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 08:41:50 2018

@author: sunkg
"""

import data
import argparse
from ResGUN_model import ResGUN
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=120,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=100,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=1000,type=int)
args = parser.parse_args()
print('Parameter assignment command.')
data.load_dataset(args.dataset,args.imgsize)
down_size = args.imgsize//args.scale
network = ResGUN(down_size,args.layers,args.featuresize,args.scale)
network.set_data_fn(data.get_batch,('train', args.batchsize,args.imgsize,down_size))
print('Go to train command.')
network.train(args.iterations, args.savedir)


