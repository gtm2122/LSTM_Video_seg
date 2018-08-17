import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image

test_img = '/data/Virginie_D/data_july_2018/data/dataset_03082018/Frames//Cre_102-1010-Ccta-10-131841/Cre_102-1010-Ccta-10-131841_Frame_image_01.bmp'

model = torch.load('FCN_trained_2.pth').cuda()

t = transforms.Compose([transforms.Grayscale(),transforms.Resize((8,8)),transforms.ToTensor()])

tensor_img = t(Image.open(test_img)).unsqueeze(0).cuda()

inp_img_np = tensor_img.cpu().numpy().squeeze()

output = model(tensor_img)

np_output = output.cpu().detach().numpy().squeeze()
import scipy.misc

scipy.misc.imsave('output_img.png',np_output)
scipy.misc.imsave('inputt_img.png',inp_img_np)