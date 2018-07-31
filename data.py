import torch
from PIL import Image
from torchvision import transforms

import os
import random

def load_data(img_dir,label_dir,batch_size,torch_transforms_img,torch_transforms_lab,rgb=False):
    
    total_batches = len(os.listdir(img_dir))//batch_size
    start = 0
    end = batch_size
    for batch_num in range(total_batches):
        
        #img_batch = torch.zeros(batch_size,2*int(rgb)+1,224,224)
        im = []
        lab= []
        
        for i in os.listdir(img_dir)[start:end]:
            im.append(Image.open(img_dir+'/'+i))
            lab.append(Image.open(label_dir+'/'+i.replace('_Frame_image_','_IMG')))
            
        
        r_num = random.randint(0,50000)
        
        random.seed(r_num)
        
        im = torch.stack([torch_transforms_img(i) for i in im])
        
        random.seed(r_num)
        
        lab = torch.stack([torch_transforms_lab(i) for i in lab])
        
        #print(im.size())
        #print(lab.size())
        
        start+=batch_size
        end+=batch_size
        
        yield(im,lab)
        
import pickle

def get_frame_num_idx(s):
    ### gets the idx where the frame number starts and where the frame number ends
    ### looks like the second underscore is where the frame identifier starts
    ud_2 = [i for i,j in enumerate(s) if j =='_'][1]
    frame_id = s[ud_2:s.find('.')]
    num_idx = [i for i,j in enumerate(frame_id) if j.isdigit()][0]
    
    frame_id = frame_id[:num_idx]
    return frame_id
    

def get_names(img_dir,label_dir,save_dir):
    dic_names_frames = {}
    
    img_frame_id = get_frame_num_idx(os.listdir(img_dir)[0])
    lab_frame_id = get_frame_num_idx(os.listdir(label_dir)[0])
    
    
    for i in os.listdir(img_dir):
        if i[:i.find(img_frame_id)] not in dic_names_frames:
            dic_names_frames[i[:i.find(img_frame_id)]] = []
        #frame_num = int(i[i.find('image_')+6:x.find('.bmp')])
        
        dic_names_frames[i[:i.find(img_frame_id)]].append((img_dir+'/'+i,label_dir+'/'+i.replace(img_frame_id,lab_frame_id)))
    
    for i in dic_names_frames:
        dic_names_frames[i] = sorted(dic_names_frames[i],\
                                     key= lambda x:int(x[0][x[0].find(img_frame_id)+len(img_frame_id):x[0].find('.')]))
    
    pickle.dump(dic_names_frames,open(save_dir,'wb'))
    
    return dic_names_frames




def sample_data_lstm(dic_obj,batch_size,seq_len,fut_seq = 0,gap=1,save_dic='pat_start_end.pkl'):
    
    
    count_dic={}
    
    for pat in dic_obj:
        count_dic[pat]=0
    
    all_names = list(dic_obj.keys())
    
    random.shuffle(all_names)
    
    used_names = []
    
    
    #count=0
    while( len(used_names) <len(all_names) ):
        
        
        new_dic = {'current':[],'future':[]}
        for _ in range(0,batch_size):
            
            
            while ( all_names[0] in used_names):
            
                random.shuffle(all_names)

                pat = all_names[0]
            print(pat)
            count = count_dic[pat]
            if(count+seq_len+fut_seq>=len(dic_obj[pat])):
                used_names.append(pat)
            else:
                
                
                
                new_dic['current'].append(dic_obj[pat][count:count+seq_len])
                new_dic['future'].append(dic_obj[pat][count+seq_len:count+seq_len+fut_seq])
                
                count_dic[pat] += gap
                
        print(count_dic)
        #yield new_dic

#def load_data_lstm(img_dir,label_dir):
    ## get the paths of images
    ## 
    
    
        