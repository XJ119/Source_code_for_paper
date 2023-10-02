from numpy.core.fromnumeric import transpose
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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


