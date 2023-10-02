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

class GCN_layer_node(nn.Module):
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


class GCN_Critical(nn.Module):
    def __init__(self, a_dim = 2, s_input_dim=4,agent_num = 7, obs_num=6):
        super(GCN_Critical, self).__init__()
        self.agent_num = agent_num
        self.obs_num = obs_num
        self.cnn_state_1 = GCN_layer_2(s_input_dim,agent_num*2)
        self.cnn_state_obs_1 = GCN_layer_2(s_input_dim,agent_num*2)
        self.mpl_state_1 = nn.Linear(s_input_dim,agent_num*2)
        self.cnn_action_1 = GCN_layer_2(a_dim,agent_num*2)

        self.attantion_layer = GCN_layer_2(agent_num*2,agent_num*2)
        self.attantion_layer_obs = GCN_layer_2(agent_num*2,agent_num*2)

        self.out1 = GCN_layer_2(1*agent_num*2*4,agent_num*2) #6为全图节点个数-1。
        self.out2 = nn.Linear(agent_num*2*agent_num,agent_num*agent_num*2) #6为全图节点个数-1。
        #self.out1.weight.data.normal_(0, 0.1)  # initialization
        self.out3 = nn.Linear(agent_num*2*agent_num,1) #6子图节点个数，1行为数。
        #self.out2.weight.data.normal_(0, 0.1)  # initialization
        #self.out_action = nn.Linear(50,1)
    
    def forward(self,adjacency,state_features,adjacency_obs,state_features_obs, action_features):
        #temp_1 = adjacency*state_features[:,0:int(self.agent_num):]
        h_s_i = F.relu(self.cnn_state_1(state_features[:,:,0:int(self.agent_num):]))
        h_s_i_obs = F.relu(self.cnn_state_obs_1(state_features_obs))
 
        h_s_e_1 = F.relu(self.mpl_state_1(state_features[:,:,int(self.agent_num):int(self.agent_num)+1])).squeeze(2)
        
        h_a_i = F.relu(self.cnn_action_1(action_features))

        #距离注意力模块部分
        attention_temp = torch.zeros(adjacency.shape[0],adjacency.shape[1],1,adjacency.shape[3]).to(device)
        attention_temp[:,:,0,:] = adjacency[:,:,0,:]

        attention_data_i = (torch.matmul(attention_temp,h_s_i)).squeeze(2)
        attention_data = F.relu(self.attantion_layer(attention_data_i))

        attention_temp_obs = adjacency_obs.unsqueeze(2)
        attention_data_i_obs = (torch.matmul(attention_temp_obs,h_s_i_obs)).squeeze(2)
        attention_data_obs = F.relu(self.attantion_layer_obs(attention_data_i_obs))

        #信息拼接
        h_s = torch.cat((attention_data,h_s_e_1 ),2)
        h_s_obs = torch.cat((h_s,attention_data_obs ),2)
        h_all = torch.cat((h_s_obs,h_a_i ),2)

        #全连接部分

        h_s_all_1 = F.relu(self.out1(h_all))
        hs_view = h_s_all_1.view(h_s_all_1.size(0), -1)
        h_s_all_3 = F.relu(self.out2(hs_view))   
        #h_a_2 = F.relu(self.out1_a(h_a_i))  
       
        
        return self.out3(h_s_all_3)

class GCN_Actor(nn.Module):
    def __init__(self, input_dim=4,sub_agent_num=4, obs_num=6):
        super(GCN_Actor, self).__init__()
        self.agent_num = sub_agent_num
        self.agent_num_obs = obs_num
        self.cnn_layer = GCN_layer_2(input_dim,sub_agent_num*2)
        self.mpl_state_1 = nn.Linear(input_dim,sub_agent_num*2)
        self.attantion_layer = GCN_layer_2(sub_agent_num*2,sub_agent_num*2)
        #self.gcn_state_1 = GCN_layer(input_dim,sub_agent_num*4)
        
        #self.gcn_state_2 = GCN_layer(sub_agent_num,sub_agent_num)
        self.cnn_layer_obs = GCN_layer_2(input_dim,sub_agent_num*2)
        self.attantion_layer_obs = GCN_layer_2(sub_agent_num*2,sub_agent_num*2)
      
        
        self.out1 = GCN_layer_2(1*sub_agent_num*2,sub_agent_num*2)
        self.out2 = nn.Linear((1+1+1)*sub_agent_num*2,sub_agent_num*sub_agent_num*2) #6子图节点个数，2行为数。
        self.out3 = nn.Linear(sub_agent_num*sub_agent_num*2,2)
    
    def forward(self,attention_weight,state_features,attention_weight_obs,state_features_obs):
        # print((attention_weight[:,0,:].permute(1,0)))    
        # print(state_features[:,0:int(self.agent_num):])
        # attention_data_temp =attention_weight[:,0,:].permute(1,0)*state_features[:,0:int(self.agent_num):]
        attention_temp = torch.zeros(attention_weight.shape[0],1,attention_weight.shape[1]).to(device)
        attention_temp[:,0,:] = attention_weight[:,0,:]
        h_t_1= F.relu(self.cnn_layer(state_features[:,0:int(self.agent_num):]))
        extr_f= state_features[:,int(self.agent_num):int(self.agent_num)+1]
        h_s_e_1 = F.relu(self.mpl_state_1(extr_f))
        #print(attention_weight[:,0,:].permute(1,0).shape)
        #if attention_weight[:,0,:].dim() ==2:
        attention_data = torch.bmm(attention_temp,h_t_1)
        #else: torch.bmm(temp_qk.permute(0,2,1),dist_V)
        #    attention_data = attention_weight[:,0,:].permute(0,2,1)*h_t_1
        h_attention = F.relu(self.attantion_layer(attention_data))
        h_all_1 = torch.cat((h_attention,h_s_e_1),1)
        #计算障碍物特征
        h_t_obs_1= F.relu(self.cnn_layer_obs(state_features_obs))
        attention_data_obs = torch.bmm(attention_weight_obs,h_t_obs_1)
        h_all = torch.cat((h_all_1,attention_data_obs),1)



        h_out1 = F.relu(self.out1(h_all))
        he_view=h_out1.view(h_out1.size(0), -1)
        h_out_2 = F.relu(self.out2(he_view))
        return torch.tanh(self.out3(h_out_2))


class GCN_Critic_DDPG(nn.Module):
    def __init__(self, a_dim = 2, s_input_dim=4,agent_num = 7, obs_num=6):
        super(GCN_Critic_DDPG, self).__init__()
        self.agent_num = agent_num
        self.obs_num = obs_num
        self.cnn_state_1 = GCN_layer_2(s_input_dim,agent_num*2)
        self.cnn_state_obs_1 = GCN_layer_2(s_input_dim,agent_num*2)
        self.mpl_state_1 = nn.Linear(s_input_dim,agent_num*2)
        self.cnn_action_1 = GCN_layer_2(a_dim,agent_num*2)

        self.attantion_layer = GCN_layer_2(agent_num*2,agent_num*2)
        self.attantion_layer_obs = GCN_layer_2(agent_num*2,agent_num*2)

        self.attantion_layer_Q = GCN_layer_2(agent_num*8,agent_num*16)
        #self.attention_layer_2 = GCN_layer_2(agent_num*8,agent_num*16)
        self.out1 = GCN_layer_2(1*agent_num*16,agent_num*8) #6为全图节点个数-1。
        self.out2 = nn.Linear(agent_num*8,agent_num*agent_num*2) #6为全图节点个数-1。
        #self.out1.weight.data.normal_(0, 0.1)  # initialization
        self.out3 = nn.Linear(agent_num*2*agent_num,1) #6子图节点个数，1行为数。
        #self.out2.weight.data.normal_(0, 0.1)  # initialization
        #self.out_action = nn.Linear(50,1)
    def graph_softmax(self, adj ,dist_Q, dist_K):
        #dist_weight = torch.zeros((adj.shape[0],self.agent_num,self.agent_num))
        Q_K = torch.matmul(dist_Q, dist_K.permute(0,2,1))/4.0
        x_exp = (torch.exp(Q_K))*adj
        x_sum = torch.sum(x_exp, dim=1)+0.00001
        #temp=x_sum.unsqueeze(2)
        dist_weight= x_exp / x_sum.unsqueeze(2)
        return dist_weight
    
    def forward(self,adjacency,adjacency_self,state_features,adjacency_obs,state_features_obs, action_features,k):
        #temp_1 = adjacency*state_features[:,0:int(self.agent_num):]
        h_s_i = F.relu(self.cnn_state_1(state_features[:,:,0:int(self.agent_num):]))

        h_s_i_obs = F.relu(self.cnn_state_obs_1(state_features_obs))
 
        h_s_e_1_1 = F.relu(self.mpl_state_1(state_features[:,:,int(self.agent_num):int(self.agent_num)+1])).squeeze(2)
        
        h_a_i_1 = F.relu(self.cnn_action_1(action_features))

        #距离注意力模块部分
        attention_temp = torch.zeros(adjacency.shape[0],adjacency.shape[1],1,adjacency.shape[3]).to(device)
        attention_temp_self = torch.zeros(adjacency.shape[0],1,adjacency.shape[3]).to(device)
        attention_temp[:,:,0,:] = adjacency[:,:,0,:]
        attention_temp_self = adjacency_self[:,k:k+1,0,:]

        attention_data_i = (torch.matmul(attention_temp,h_s_i)).squeeze(2)    
        attention_data_1 = F.relu(self.attantion_layer(attention_data_i))


        attention_temp_obs = adjacency_obs.unsqueeze(2)
        attention_data_i_obs = (torch.matmul(attention_temp_obs,h_s_i_obs)).squeeze(2)
        attention_data_obs_1 = F.relu(self.attantion_layer_obs(attention_data_i_obs))

        h_all_dist_attention_1 = torch.cat((attention_data_1,attention_data_obs_1 ),2)
        h_all_dist_attention_2 = torch.cat((h_all_dist_attention_1,h_s_e_1_1 ),2)
        h_all_dist_attention_3 = torch.cat((h_all_dist_attention_2,h_a_i_1 ),2)

        self_dist_attention_q = F.relu(self.attantion_layer_Q(h_all_dist_attention_3))

        #temp_qk = self.graph_softmax(attention_temp_t_k,self_dist_attention_q, self_dist_attention_k)
        h_attention_self = torch.matmul(attention_temp_self,self_dist_attention_q)
        #attention_data = F.relu(self.attention_layer_2(h_attention_self))
        #全连接部分

        h_s_all_1 = F.relu(self.out1(h_attention_self))
        hs_view = h_s_all_1.view(h_s_all_1.size(0), -1)
        h_s_all_3 = F.relu(self.out2(hs_view))   
        #h_a_2 = F.relu(self.out1_a(h_a_i))  
       
        
        return self.out3(h_s_all_3)

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


