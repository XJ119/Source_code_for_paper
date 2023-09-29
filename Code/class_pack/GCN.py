import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
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
    def forward(self,adjacency,input_feature):
        support = torch.matmul(input_feature, self.weight)
        output =  torch.matmul(adjacency, support)
        if self.use_bias:
            output= output+self.bias
        return output
class GCN_layer_2(nn.Module):
    def __init__(self, input_dim, output_dim,use_bias=True):
        super(GCN_layer_2, self).__init__()
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


class GCN_Critical(nn.Module):
    def __init__(self, a_dim = 2, s_input_dim=5,agent_num = 7):
        super(GCN_Critical, self).__init__()
        self.agent_num = agent_num
        self.gcn_state_1 = GCN_layer(s_input_dim,8)
        self.gcn_action_1 = GCN_layer(a_dim,8)
        self.gcn_adj_1 = GCN_layer_2(agent_num,8)
        #self.gcn_action_1 = nn.Linear(a_dim*agent_num,5*agent_num) #6子图节点个数，1行为数。
        #self.gcn_action_1.weight.data.normal_(0, 0.1)  # initialization

        #self.gcn_state_2 = GCN_layer(8,8)
        #self.gcn_action_2 = GCN_layer(8,8)

        self.out1 = nn.Linear(1*agent_num*8*3,agent_num*8) #6为全图节点个数-1。
        self.out1.weight.data.normal_(0, 0.1)  # initialization
        self.out2 = nn.Linear(agent_num*8,1) #6子图节点个数，1行为数。
        self.out2.weight.data.normal_(0, 0.1)  # initialization
        #self.out_action = nn.Linear(50,1)
    
    def forward(self,adjacency,state_features, action_features):
        h_s_1 = F.relu(self.gcn_state_1(adjacency*0.0+torch.eye(self.agent_num).to('cuda'),state_features))
        #h_s_1= F.normalize(h_s_1)
        #ha_input=action_features.view(action_features.size(0), -1)
        h_a_1 = F.relu(self.gcn_action_1(adjacency*0.0+torch.eye(self.agent_num).to('cuda'),action_features))
        h_d_1 = F.relu(self.gcn_adj_1(adjacency))
        #h_a_1= F.normalize(h_a_1)
        #h_s_2= F.leaky_relu(self.gcn_state_2(adjacency,h_s_1),negative_slope=0.02)
        #h_s_2= F.normalize(h_s_2)
        #h_a_2= F.leaky_relu(self.gcn_action_2(adjacency,h_a_1),negative_slope=0.02)
        #h_a_2= F.normalize(h_a_2)
        hs_view=h_s_1.view(h_s_1.size(0), -1)
        ha_view=h_a_1.view(h_a_1.size(0), -1)

        hadj_view = h_d_1.view(h_d_1.size(0), -1)

        h_view_temp = torch.cat((hs_view,ha_view),1) 
        h_view = torch.cat((h_view_temp,hadj_view),1)
        h_out1 = F.relu(self.out1(h_view))
        return self.out2(h_out1)

class GCN_Actor(nn.Module):
    def __init__(self, input_dim=5,sub_agent_num=4):
        super(GCN_Actor, self).__init__()
        self.gcn_state_1 = GCN_layer(input_dim,8)

        #self.gcn_state_2 = GCN_layer(8,8)
        self.out1 = nn.Linear(1*sub_agent_num*8,sub_agent_num*8)
        self.out2 = nn.Linear(sub_agent_num*8,2) #6子图节点个数，2行为数。
    
    def forward(self,adjacency,state_features):
        h_s_1= F.relu(self.gcn_state_1(adjacency,state_features))

        #h_s_2= F.leaky_relu(self.gcn_state_2(adjacency,h_s_1),negative_slope=0.02)

        hs_view=h_s_1.view(h_s_1.size(0), -1)
        h_out1 = F.relu(self.out1(hs_view))

        return torch.tanh(self.out2(h_out1))

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


