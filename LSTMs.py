import torch.nn as nn
import torch
from torch.autograd import Variable

class ec(nn.Module):
    def __init__(self,input_size = 1,batch_size=32,\
                 hidden_size = 64,num_layers=1,bias=False,\
                 bidirectional=False,seq_len=2,ec_flag=True):
        super().__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        self.bias = bias
        self.bidirectional = bidirectional
        
        self.inp_layer = nn.Linear(self.input_size,self.input_size) ### buffer transform the input
        
        self.lstm1 = nn.LSTM(input_size = self.input_size,\
                             hidden_size = self.hidden_size,num_layers=self.num_layers,\
                             bias=self.bias,bidirectional=self.bidirectional)
        self.num_dir = 2 if self.bidirectional else 1
        #self.cl = nn.Linear(seq_len*hidden_size,3)
        self.hidden = self.init_hidden()
        #self.num_dir = num_dir
        self.ec_flag = ec_flag
        self.dec2lin = nn.Linear(hidden_size,input_size) ## buffer to transform the output
        
    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers*self.num_dir,self.batch_size,self.hidden_size)),\
                Variable(torch.zeros(self.num_layers*self.num_dir,self.batch_size,self.hidden_size)))
                
        
    def forward(self,x,h):
        
        orig_size = x.size()
        
        x = x.view(orig_size[0]*orig_size[1],-1)
        
        x = self.inp_layer(x)
        
        x = x.view(orig_size[0],orig_size[1],-1)
        
        x,h = self.lstm1(x,h)
        
        if(not self.ec_flag):
            #print(x.size())
            x = x.view(x.size(0)*x.size(1),-1)
            #print(x.size())
            return self.dec2lin(x)
        #print('x',x.size())
        #print(len(h))
        #print(h[0].size())
        #print(len(h))
        #x = x.view(-1,x.size(0)*x.size(2))
        #print(x.size())
        #print(x.size())
        #x = self.cl(x)
        
        return x,h
    


