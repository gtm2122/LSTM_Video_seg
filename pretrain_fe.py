# pretraining feature extractors with CT images

# input = original image, output = original image OR gt for label
from torchvision import transforms
import torch
from PIL import Image
from torch.autograd import Variable

def train_fe(m,dic,optimizer,im_size=256,train=True):
    if train:
        m.train()
    else:
        m.eval()
    criterion = torch.nn.MSELoss()
    
    current_batch = dic['current']
    future_batch = dic['future']
    
    
    t = transforms.Compose([transforms.Grayscale(),transforms.Resize((im_size,im_size)),transforms.ToTensor()])
    
    
    #img_size_buf
    current_tensor = torch.zeros(len(current_batch)*len(current_batch[0]),1,im_size,im_size)
    future_tensor = torch.zeros(len(future_batch)*len(future_batch[0]),1,im_size,im_size)
    
    future_tensor = torch.cat((future_tensor,current_tensor),dim=0)
    cc = 0
    
    optimizer.zero_grad()
    
    for c_seq,f_seq in zip(current_batch,future_batch):
        ### seq is the sequence of addresses in a batch
        #print("f_seq")
        #print("c_seq")
        #print(f_seq)
        #print(c_seq)
        for f_img in f_seq:
            #print(f_seq)
            #print(f_img)
            img_tens = t(Image.open(f_img)).cuda()
            
            future_tensor[cc,:,:,:] = img_tens
            cc+=1
        
        for c_img in c_seq:
            #print(f_img)
            img_tens = t(Image.open(c_img)).cuda()
            future_tensor[cc,:,:,:] = img_tens
            cc+=1
            
    inputs = Variable(future_tensor.cuda())
    labels = Variable(future_tensor.cuda())
    
    
    outputs = m(inputs)
    
    #print(inputs.size())


    loss = criterion(outputs,labels)
    
    if train:
        loss.backward()
        optimizer.step()
    
    return m,loss.item(),optimizer
import pickle
from FCN8 import FCN8_comb        
m = FCN8_comb().cuda()         
            
opt = torch.optim.Adam(m.parameters(),lr=0.002)
from data import sample_data_lstm

dic_train= pickle.load(open('./dic_train_lstm.pkl','rb'))
dic_test= pickle.load(open('./dic_test_lstm.pkl','rb'))
dic_val= pickle.load(open('./dic_val_lstm.pkl','rb'))
train_loss_epoch = []
val_loss_epoch = []

dic_dic = {'train':dic_train,'val':dic_val}

best_train_loss = 1e8
best_val_loss = 1e8

epochs = 200

for epochs in range(epochs):
    #for phase in ['val','train']:
    for phase in ['train','val']:
   
        train_loss=0
        val_loss = 0
        
        for i in sample_data_lstm(dic_dic[phase],2,4,4,gap=3,save_dic = './dic_test_lstm.pkl'):
            #print(i)
            dic=i
            #print("dic")
            #print(dic)
            m,loss_batch,opt=train_fe(m,dic,opt,im_size=256,train=True)
            
            if phase =='train':
                train_loss+=loss_batch
               
            else:
                val_loss+=loss_batch
                 
            #break
        if val_loss < best_val_loss:
            best_model = m
            best_val_loss =val_loss
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)
        
        #break
    #break
    
#print(train_loss_epoch)
#print(val_loss_epoch)
torch.save(m.cpu(),'FCN_trained_'+str(epochs)+'.pth')