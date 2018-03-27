from model_lr_decay import EDSR
from data import load_dataset,crop_center,get_batch,get_test_set

load_dataset("D://LAB//master_thesis//EDSR-Tensorflow//dataset//General-100",100)
network=EDSR(50,16,64,2) #ONE BASELINE
network.set_data_fn(get_batch,(10,100,50),get_test_set,(100,50))
network.train()
