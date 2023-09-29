from numpy.core.fromnumeric import transpose
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class GCN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,use_bias=True):
        super(GCN_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias= nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    def forward(self,adjacency):
        #support = torch.matmul(input_feature, self.weight)
        output =  torch.matmul(adjacency, self.weight)
        if self.use_bias:
            output= output+self.bias
        return output
class RNN_net(nn.Module):
    def __init__(self, state_dim=3,input_dim=3,out_dim=3):
        super(RNN_net, self).__init__()
        self.state_net = nn.Linear(state_dim,64)
        self.input_net = nn.Linear(input_dim,64)
        self.layer_1 = nn.Linear(128,128)
        self.layer_2 = nn.Linear(128,64)
        self.layer_3 = nn.Linear(64,out_dim)
    
    def forward(self,train_state_data,train_input_data):
        h_s_1= F.relu(self.state_net(train_state_data))
        h_i_1= F.relu(self.input_net(train_input_data))
        h_all = torch.cat((h_s_1,h_i_1),1)
        h_out1 = F.relu(self.layer_1(h_all))
        h_out2 = F.relu(self.layer_2(h_out1))

        return torch.tanh(self.layer_3(h_out2))
    
class RNN_net_nostate(nn.Module):
    def __init__(self, state_dim=3,input_dim=3,out_dim=3):
        super(RNN_net_nostate, self).__init__()
        self.state_net = nn.Linear(state_dim,64)
        self.input_net = nn.Linear(input_dim,64)
        self.layer_1 = nn.Linear(64,128)
        self.layer_2 = nn.Linear(128,64)
        self.layer_3 = nn.Linear(64,out_dim)
    
    def forward(self,train_state_data,train_input_data):
        #h_s_1= F.relu(self.state_net(train_state_data))
        h_i_1= F.relu(self.input_net(train_input_data))
        #h_all = torch.cat((h_s_1,h_i_1),1)
        h_out1 = F.relu(self.layer_1(h_i_1))
        h_out2 = F.relu(self.layer_2(h_out1))

        return torch.tanh(self.layer_3(h_out2))

class RNN_net_2(nn.Module):
    def __init__(self, state_dim=3,input_dim=3,out_dim=3):
        super(RNN_net_2, self).__init__()
        self.state_net = nn.Linear(state_dim,64)
        self.input_net = nn.Linear(input_dim,64)
        self.layer_0 = nn.Linear(128,256)
        self.layer_1 = nn.Linear(256,128)
        self.layer_2 = nn.Linear(128,64)
        self.layer_3 = nn.Linear(64,out_dim)
    
    def forward(self,train_state_data,train_input_data):
        h_s_1= F.relu(self.state_net(train_state_data))
        h_i_1= F.relu(self.input_net(train_input_data))
        h_all = torch.cat((h_s_1,h_i_1),1)
        h_out0 = F.relu(self.layer_0(h_all))
        h_out1 = F.relu(self.layer_1(h_out0))
        h_out2 = F.relu(self.layer_2(h_out1))

        return torch.tanh(self.layer_3(h_out2))

class RNN_net_AT(nn.Module):
    def __init__(self, state_dim=3,input_dim=3,out_dim=3):
        super(RNN_net_AT, self).__init__()
        self.Lrelu = nn.LeakyReLU(0.01)
        self.state_net = GCN_layer(state_dim,64)
        self.input_net = GCN_layer(input_dim,64)
        self.attention_net_1 = GCN_layer(64,64)
        self.attention_net_2 = GCN_layer(64,1)
    
        self.attention_net_v = GCN_layer(64,64)

        self.layer_1 = nn.Linear(128,128)
        self.layer_2 = nn.Linear(128,64)
        self.layer_3 = nn.Linear(64,out_dim)
    def softmax(self, x):
        x_exp = (torch.exp(x))
        x_sum = torch.sum(x_exp, dim=1)
        #temp=x_sum.unsqueeze(2)
        at_weight= x_exp / x_sum.unsqueeze(2)
        return at_weight.permute(0,2,1)

    def forward(self,train_state_data,train_input_data):
        h_s_1= self.Lrelu(self.state_net(train_state_data))
        s_value_1 = self.Lrelu(self.attention_net_1(h_s_1))
        s_rate = self.Lrelu(self.attention_net_2(s_value_1))
        s_rate_norm = self.softmax(s_rate)
        attention_hs = (torch.matmul(s_rate_norm ,s_value_1)).squeeze(1)

        h_i_1= self.Lrelu(self.input_net(train_input_data))
        h_all = torch.cat((attention_hs,h_i_1),1)
        h_out1 = self.Lrelu(self.layer_1(h_all))
        h_out2 = self.Lrelu(self.layer_2(h_out1))

        return torch.tanh(self.layer_3(h_out2))
    

class RNN_net_AT_2(nn.Module):
    def __init__(self, state_dim=3,input_dim=3,out_dim=3,len =2 ):
        super(RNN_net_AT_2, self).__init__()
        self.time_len = len
        self.Lrelu = nn.LeakyReLU(0.01)
        self.state_net = GCN_layer(state_dim,64)
        self.input_net = GCN_layer(input_dim,64)
        self.attention_net_1 = GCN_layer(64,256)
        self.attention_net_2 = GCN_layer(256,1)
    
        self.attention_net_v = GCN_layer(256,64)

        self.layer_1 = nn.Linear(128+64,128)
        self.layer_2 = nn.Linear(128,64)
        self.layer_3 = nn.Linear(64,out_dim)
    def softmax(self, x):
        x_exp = (torch.exp(x))
        x_sum = torch.sum(x_exp, dim=1)
        #temp=x_sum.unsqueeze(2)
        at_weight= x_exp / x_sum.unsqueeze(2)
        return at_weight.permute(0,2,1)

    def forward(self,train_state_data,train_input_data):
        
        h_s_1= self.Lrelu(self.state_net(train_state_data))
        last_data = h_s_1[:,self.time_len-1]
        time_data = h_s_1[:,0:self.time_len-1]

        s_value_1 = self.Lrelu(self.attention_net_1(time_data))
        s_rate = self.Lrelu(self.attention_net_2(s_value_1))
        s_rate_norm = self.softmax(s_rate)
        attention_hs = (torch.matmul(s_rate_norm ,s_value_1)).squeeze(1)
        attention_hs_out = self.attention_net_v(attention_hs)

        h_i_1= self.Lrelu(self.input_net(train_input_data))
        h_all = torch.cat((attention_hs_out,h_i_1),1)
        h_all_2 = torch.cat((h_all ,last_data),1)
        h_out1 = self.Lrelu(self.layer_1(h_all_2))
        h_out2 = self.Lrelu(self.layer_2(h_out1))

        return torch.tanh(self.layer_3(h_out2))


if __name__ == '__main__':
    #GCN_net1=GCN_NET()
    # a1=np.array([[1,2,3,4],[5,6,7,8],[11,12,13,14],[2,3,4,5]])
    # b1=torch.from_numpy(a1).unsqueeze(0)
    # print(b1)
    # c1=np.array([6,6,6,6])
    # d1=torch.from_numpy(c1)
    # print(d1)
    # print(b1+c1)
    # #print(np.normalize_axis_index)
    # print(5%2)
    data=np.array([1,2,3])
    
    print('data:',data)
    tensor_data= torch.from_numpy(data)#.to(device)
    data2=tensor_data.unsqueeze(1)
    print('tensor_data::',tensor_data)


