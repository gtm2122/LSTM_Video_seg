test_dir_img = '/data/Virginie_D/Only_CTScore/Only_CTScore/test/val/img/W/' 
test_dir_lab = '/data/Virginie_D/Only_CTScore/Only_CTScore/test/val/gt/W/' 
import os
import torch
from train_test import *
#from Unet2 import *
os.environ['CUDA_VISIBLE_DEVICES']='1'
sz = (64,64)
name_model='unet128_aug2_bce_128x128_transp3_68BEST.pth'
test_net(name_model,test_dir_img,test_dir_lab,False,sz,sz,False)