from data import *
from unet import *
from torchvision import transforms
### train the model
import Augmentor
import os
from torch.nn import init
p = Augmentor.Pipeline()
from torch.autograd import Variable
from dice_loss import *
#name_model = 'saved_unet_pad_augm'
import matplotlib.pyplot as plt
from train_test import *

###### USED WITH BCE LOSS


from data import *
from Unet2 import *
from torchvision import transforms
### train the model
import Augmentor
import os
p = Augmentor.Pipeline()
from torch.autograd import Variable
from dice_loss import *
#name_model = 'saved_unet_pad_augm'
import matplotlib.pyplot as plt
from train_test import *

###### USED WITH BCE LOSS


train_dir_lab = '/data/Virginie_D/Only_CTScore/Only_CTScore/data/train/gt/W/'
train_dir_img = '/data/Virginie_D/Only_CTScore/Only_CTScore/data/train/img/W/'

val_dir_img = '/data/Virginie_D/Only_CTScore/Only_CTScore/data/val/img/W/'
val_dir_lab = '/data/Virginie_D/Only_CTScore/Only_CTScore/data/val/gt/W/'

test_dir_img = '/data/Virginie_D/Only_CTScore/Only_CTScore/test/val/img/W/' 
test_dir_lab = '/data/Virginie_D/Only_CTScore/Only_CTScore/test/val/gt/W/' 

p.random_distortion(probability=0.9,grid_width=8,grid_height=8,magnitude=5)
#p.random_distortion(probability=1,grid_width=6,grid_height=6,magnitude=9)

#p.gaussian_distortion(probability=1,grid_width=3,grid_height=3,magnitude=8)
#p.rotate(probability=0.6,max_left_rotation=3,max_right_rotation=3)
#p.shear(probability=0.6,max_shear_left=2,max_shear_right=2)

sz = (256,256)
torch_tf_aug = transforms.Compose([p.torch_transform(),transforms.Resize(sz)\
                                   ,transforms.RandomVerticalFlip(p=0.6),\
                                   transforms.RandomRotation(degrees=3),
                                   transforms.RandomAffine(degrees=1,translate=(0.02,0.02)),
                                   transforms.RandomHorizontalFlip(p=0.6),transforms.Grayscale(),
                                   transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])
                                   
                                  ])


torch_tf_aug_lab = transforms.Compose([p.torch_transform(),transforms.Resize(sz)\
                                   ,transforms.RandomVerticalFlip(p=0.6),\
                                   transforms.RandomRotation(degrees=3),
                                   transforms.RandomAffine(degrees=1,translate=(0.02,0.02)),
                                   transforms.RandomHorizontalFlip(p=0.6),transforms.Grayscale(),
                                   transforms.ToTensor()])
                                  

torch_tf_non_aug = transforms.Compose([transforms.Resize(sz),transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])
                                       ])

torch_tf_non_aug_lab = transforms.Compose([transforms.Resize(sz),transforms.Grayscale(),transforms.ToTensor()
                                       ])


tf_dic = {'train':[(torch_tf_aug,torch_tf_aug_lab),(torch_tf_non_aug,torch_tf_non_aug_lab)],\
    'val':[(torch_tf_non_aug,torch_tf_non_aug_lab)]}


tf_dic_non_aug = {'train':[(torch_tf_non_aug,torch_tf_non_aug_lab)],\
    'val':[(torch_tf_non_aug,torch_tf_non_aug_lab)]}


data_dirs = {'train':[train_dir_img,train_dir_lab],'val':[val_dir_img,val_dir_lab]}

#seg_model = Unet(num_layers=5,padding=1,transp=True,num_classes=1,pad_center=True)

#seg_model=seg_model.cuda()

#criterion = torch.nn.BCELoss()
#criterion = torch.nn.MSELoss()
#criterion=criterion.cuda()

#crit_list = [torch.nn.BCELoss(),torch.nn.MSELoss(),torch.nn.BCEWithLogitsLoss()]
criterion = torch.nn.BCELoss()
#name_model='bcelog'


seg_model = UNet10243((1,sz[0],sz[1]))
#seg_model = UNet1024((1,sz[0],sz[1]))

seg_model=seg_model.cuda()
init_weight(seg_model)

name_model='unet1024_aug2_bce_256x256_transp3_'
#if(pad_bool==False):

train_net(seg_model,name_model,num_epochs=250,lr=0.001,criterion=criterion,\
          tf_dic=tf_dic,data_dirs=data_dirs,smax=False,batch_size=8)
#test_net(name_model,smax=True,img_size=(256,256),lab_size=(196,196),show_image=False)
