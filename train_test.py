from torch.nn import functional as F
from data import *
from unet import *
from torchvision import transforms
### train the model
import Augmentor
import os
#p = Augmentor.Pipeline()
from torch.autograd import Variable
from dice_loss import *
#name_model = 'saved_unet_pad_augm'
import matplotlib.pyplot as plt
#a = torch.ones((4,1,256,256)).cuda()

#output_shape = seg_model(a).size()

from torch import optim

def train_net(seg_model,name_model,num_epochs,lr,criterion,tf_dic,data_dirs,smax=True,batch_size=4):
    best_dice = 0
    best_epoch = 0
    train_loss=[]
    val_loss=[]
    
    best_epochs=0
    op = optim.Adam(seg_model.parameters(),lr=lr)
    try:
        for epochs in range(num_epochs):

            '''
            if(epochs>0 and epochs%10==9):

                torch.save(seg_model,name_model+str(epochs)+'.pth')  
            '''
            for phase in ['train','val']:
                running_loss = 0
                running_score=0
                count=0
                for tf in tf_dic[phase]:
                    for img,label in load_data(img_dir=data_dirs[phase][0],\
                                               label_dir=data_dirs[phase][1],\
                                               batch_size=batch_size,torch_transforms_img=tf[0],\
                                               torch_transforms_lab=tf[1]):
                        #lab = label.type(torch.LongTensor)
                        op.zero_grad()
                        #print(label[0,0,:,:])
                        img = Variable(img).cuda()
                        lab = Variable(label).cuda()
                        lab[lab>0]=1.0
                        if(phase=='train'):
                            img.requires_grad = True
                            lab.requires_grad = False
                            seg_model.train()
                        else:
                            img.requires_grad = False
                            lab.requires_grad = False
                            seg_model.eval()
                        #print(img.size())    
                        output = seg_model(img)
                        pred_mask=output
                        if('bcelog' not in name_model):
                            if smax:
                                pred_mask = F.softmax(output.view(batch_size,-1),dim=1)
                            else:
                                pred_mask = F.sigmoid(output)
                        #print(pred_mask.size())
                        #if(isinstance(criterion,object)):

                        loss = criterion(pred_mask.view(-1),lab.squeeze().view(-1))
                        #print(loss)
                        if(phase=='train'):
                            loss.backward()
                            op.step()

                        running_loss+=loss.item()  
                        pred_mask=(pred_mask>0.5*pred_mask.max()).float()
                        #break
                        count+=1

                        running_score+= dice_coeff(pred_mask,lab).item()
                        del(pred_mask)
                        #print(running_score)
                        del(img)
                        del(label)

                if(phase=='train'):
                    print('TRAIN')
                    print('epoch ',epochs)

                    train_loss.append(running_loss)
                    print('loss ',train_loss[-1])

                    plt.plot(train_loss)
                    plt.savefig('./'+str(name_model)+'train_loss.png')
                    print('avg_dice ',running_score/count)

                    plt.close()
                else:
                    if running_score/count>best_dice:
                        best_model = seg_model
                        torch.save(best_model,name_model+str(best_epochs)+'BEST.pth')
                        best_dice=running_score/count
                        best_epochs=epochs
                    print('VAL')
                    print('epoch ',epochs)
                    print('avg_dice ',running_score/count)
                    val_loss.append(running_loss)
                    print('loss ',val_loss[-1])

                    plt.plot(val_loss)
                    plt.savefig('./'+str(name_model)+'val_loss.png')
                    plt.close()

        torch.save(seg_model,name_model+str(epochs)+'LAST.pth') 
        #torch.save(best_model,name_model+str(best_epochs)+'.pth') 
        return seg_model
    except KeyboardInterrupt:
        torch.save(seg_model,name_model+str(epochs)+'LAST.pth') 
        torch.save(best_model,name_model+str(best_epochs)+'NEST.pth') 
        return seg_model
        sys.exit()
       
from dice_loss import dice_coeff
import numpy as np
import matplotlib.pyplot as plt
      
def test_net(name_model,test_dir_img,test_dir_lab,smax=False,img_size=(256,256),lab_size=(252,252),show_image=False):
    
    torch_tf_non_aug = transforms.Compose([transforms.Resize(img_size),transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])
                                           ])

    torch_tf_non_aug_lab = transforms.Compose([transforms.Resize(lab_size),transforms.Grayscale(),transforms.ToTensor()
                                           ])

    
    seg_model = torch.load(name_model)
    seg_model.cuda()
    seg_model.eval()
    batch_size=4
    avg_dice = 0
    count = 0
    for img,label in load_data(img_dir=test_dir_img,\
                                       label_dir=test_dir_lab,\
                                       batch_size=batch_size,torch_transforms_img=torch_tf_non_aug,\
                                       torch_transforms_lab=torch_tf_non_aug_lab):



        img=Variable(img.cuda(),requires_grad=False)
        output=seg_model(img)
        label[label>0]=1
        if(not smax):
            output=nn.functional.sigmoid(output)
            pred_mask=(output>0.5*output.max()).float()
        else:
            output=nn.functional.softmax(output)
            pred_mask=(output>0.5*output.max()).float()
        c=0

        
        avg_dice +=dice_coeff(pred_mask,Variable(label).cuda()).item()
        if(show_image):
            for i in output:
                t_lab = label.squeeze().numpy()

                #print(t_lab.shape)
                i=i.cpu().squeeze().detach()
                imag = i.numpy()

                seg_result = np.zeros_like(imag)
                seg_result[imag>0.5*imag.max()]=255

                print(imag.shape)
                plt.imshow(imag),plt.show()
                plt.imshow(seg_result),plt.show()
                plt.imshow(t_lab[c,:,:]),plt.show()
                plt.close()
                c+=1
        count+=1
        del img
        del output

    print(avg_dice/count)
    