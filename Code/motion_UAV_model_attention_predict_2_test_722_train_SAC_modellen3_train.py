from operator import index
import time
import numpy as np
import math
import random
from torch.autograd import Variable
from collections import namedtuple
import matplotlib.pyplot as plt
import class_pack.anti_function as AFn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import class_pack.GCN_4_obs_DDPG_v3_dist_attation_reward as GCN
import math
import os
import RNN
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import class_pack.ReplayMem as Mem
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
array_tran=T.ToTensor()
writer = SummaryWriter("flocking_run\experment_interprupt_attention_UAV_model_2_predict_SAC_train_3_test")#_6 0.012
Transition = namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_self','dist_weight_obs','action', 'next_state_o','next_state_o_obs','next_dist_weight','next_dist_weight_self','next_dist_weight_obs', 'reward'))
Transition_predicate = namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_obs','action'))
#Initialize environment parameters
num_agents = 6
num_obs = 3
sub_graph_size = 6
t_gap = 0.5
r_c = 30.0
r_o = 15.0
d_o =10.0
v_max=1
v_des=0.5
d_des=5
d_graph=d_des*2.0

c_alpha_p=0.03
c_alpha_v = 1.5
c_beta_p = 0.2
c_beta_v = 0.8
c_gamma_p = 0.03
c_gamma_v = 2.0
c_obs = 1.5
#Building agent location and speed
x = np.zeros((num_agents,2))
x_1 = np.zeros((num_agents,2))
v = np.zeros((num_agents,2))
v_1 = np.zeros((num_agents,2))
#v_action = np.zeros((num_agents,2))
v_action = np.zeros((num_agents,2))
v_action_store = np.zeros((num_agents,2))

#Initialize Flocking Parameters
Anti_F = AFn.Anti_Fn()  
encode_bit = 4
#Training Parameters
capacity_size = 30000
epsilon = 0.3
BATCH_SIZE = 512
BATCH_SIZE_P = 1024
GAMMA_1 = 0.0
GAMMA_2 = 0.0
EPS_START = 0.0
EPS_END = 0.00
EPS_DECAY = 80000
TARGET_UPDATE = 5
TAU = 0.05      # soft replacement
TAU_A = 0.002
TAU_P = 0.2
plot_flag = False
eval_capacity = 50
point = 0
action=[]



global_loss = None


for s_k in range(num_agents):
    
    action.append(None)
    #state_o.append(None)
    #global_loss.append(0)

num_episodes= 4001
Agent_Mem_H=Mem.ReplayMemory(int(capacity_size))
Agent_Mem_M=Mem.ReplayMemory(int(capacity_size/3))
Agent_Mem_L=Mem.ReplayMemory(int(capacity_size/3))

Agent_Mem_predicate = Mem.Predict_Memory(int(capacity_size*3))


#DDPG action
GCN_critic_DDPG_policy= GCN.GCN_Critic_DDPG(2,4,num_agents,num_obs).to(device)  #action， state
#GCN_critical_policy2= torch.load('GCN_critical_target1_v4_y9500.pth').to(device)
GCN_critical_DDPG_target= GCN.GCN_Critic_DDPG(2,4,num_agents,num_obs).to(device) 
GCN_critical_DDPG_target.load_state_dict(GCN_critic_DDPG_policy.state_dict())
GCN_critical_DDPG_target.eval()
GCN_critical_DDPG_optimizer = optim.Adam(GCN_critic_DDPG_policy.parameters(),lr=0.0001)

GCN_critic_DDPG_policy_2= GCN.GCN_Critic_DDPG(2,4,num_agents,num_obs).to(device)  #action， state
#GCN_critical_policy2= torch.load('GCN_critical_target1_v4_y9500.pth').to(device)
GCN_critical_DDPG_target_2= GCN.GCN_Critic_DDPG(2,4,num_agents,num_obs).to(device) 
GCN_critical_DDPG_target_2.load_state_dict(GCN_critic_DDPG_policy_2.state_dict())
GCN_critical_DDPG_target_2.eval()
GCN_critical_DDPG_optimizer_2 = optim.Adam(GCN_critic_DDPG_policy_2.parameters(),lr=0.0001)


#Loading motion model
time_len = 3 #sequence length
motion_model = torch.load('train_motion_model_uav_412_attention_2Hz_len3_21000.pth',map_location=device)
#Policy model
GCN_actor_policy= GCN.GCN_Actor(4,sub_graph_size).to(device)
#GCN_actor_policy = torch.load('GCN_actor_target_v4_3_3_UAV_pre_para2_5_0_entropy_15_model3_2_124000.pth',map_location=device)
GCN_actor_target= GCN.GCN_Actor(4,sub_graph_size).to(device)
GCN_actor_target.load_state_dict(GCN_actor_policy.state_dict())

GCN_actor_target.eval()
GCN_actor_optimizer = optim.Adam(GCN_actor_policy.parameters(),lr=0.00005)

# Behavior reasoning model
GCN_actor_predicate= GCN.GCN_Actor(4,sub_graph_size).to(device)
GCN_actor_predicate.load_state_dict(GCN_actor_policy.state_dict())
#GCN_actor_predicate = torch.load('GCN_predicate_target1_v4_3_3_UAV_pre_para2_5_0_entropy_15_model3_2_124000.pth',map_location=device)
GCN_predicate_optimizer = optim.Adam(GCN_actor_predicate.parameters(),lr=0.00005)

#Entropy 
target_entropy = -1.2  # The best value of target entropy if from 0.8 to 2.0
log_alpha = torch.zeros(1, requires_grad=True,device=device)
alpha = log_alpha.exp()
optimizer_alpha = torch.optim.Adam([log_alpha], lr=0.00005)


#noise
var=0.0
sigma_c = 5
run_step=0
using_flocking_flag=False
plt.ion()


#Function

def adjacency_matrix_sample(total_num,sample_index):
    sample_num=sample_index.shape[0]
    init_sample_matrix= np.zeros((total_num, sample_num))
    for col_sample in range(sample_num):
        init_sample_matrix[sample_index[col_sample],col_sample]=1
    return init_sample_matrix


def tanh_function_followers(data,d_quan):
    data = data*d_graph
    if data >= 0:
        loss = (data/1.2)*(data/1.2)*(data/1.2)*(data/1.2)

        return loss
    else:
        loss = 1.0*(data/1.0)*(data/1.0)*(data/1.0)*(data/1.0)
        #loss = math.tan(math.pi*0.0001+(math.pi/(2*d_quan))*data)*(-1.0)
        if loss<0.5:
            loss = 0 
        return loss

def tanh_function_leader(data,d_quan):
    data = data*d_graph
    if data >= 0:
        loss = (data/2.0)*(data/2.0)*(data/2.0)*(data/2.0)
        
        return loss
    else:
        loss = 0
        #loss = math.tan(math.pi*0.0001+(math.pi/(2*d_quan))*data)*(-1.0)
        if loss<0.5:
            loss = 0 
        return loss



def Loss_function_algorithm2_local(pos,vel,index,d_des,d_graph,adj_num,adj_matrix,extra_data,adj_obs_num,vel_a):
    global c_alpha_p
    global c_alpha_v
    global c_gamma_p
    global c_gamma_v
    #i_index=np.where(index==i)
    dis_err_sum=0
    vel_err_sum=0
    vel_err_sum_2=0
    vel_err_2 = 0
    d_quan= (d_des/d_graph)
    #td_xr = np.zeros((2,1))
    #td_vr = np.zeros((2,1))


    for j in range(sub_graph_size):
        node_quan_pos = pos[j,:]
        node_quan_vel = vel[j,:]
        node_quan_vel_a = vel_a[j,:]

        temp_d = node_quan_pos- pos[0,:]
        temp_v = node_quan_vel- vel[0,:]
        temp_v_a = node_quan_vel_a- vel_a[0,:]

        if j != 0:
            cal_flag=1
                #nbr_count=nbr_count+1
        else:    
            cal_flag=0
            #位置误差
        # if  index[j] == 0:   
        #     dis_err= 0#tanh_function_leader(np.linalg.norm(temp_d)-d_quan, d_quan)*cal_flag
        #     vel_err = 0#1.0 * math.sqrt((np.linalg.norm(temp_v))**2)*cal_flag
        # else:
        if adj_obs_num == 0:
            dis_err= tanh_function_followers(np.linalg.norm(temp_d)-d_quan, d_quan)*cal_flag*adj_matrix[0,j]
            vel_err = math.sqrt((np.linalg.norm(temp_v))**2)*cal_flag*adj_matrix[0,j]
            
        else:
            dis_err= 0.8*tanh_function_followers(np.linalg.norm(temp_d)-d_quan, d_quan)*cal_flag*adj_matrix[0,j]
            vel_err = 0.2*math.sqrt((np.linalg.norm(temp_v))**2)*cal_flag*adj_matrix[0,j]
        
        vel_err_2 = math.sqrt((np.linalg.norm(temp_v_a))**2)*cal_flag

        dis_err_sum = dis_err_sum + dis_err

        
        vel_err_sum = vel_err_sum + vel_err
        vel_err_sum_2 = vel_err_sum_2 + vel_err_2
    if adj_num ==1:
        dis_err_sum = 625
        vel_err_sum =1.0

    loss_alpha= 0.8*c_alpha_p*(dis_err_sum) + 0.8*c_alpha_v*vel_err_sum
    #print(extra_data[0,0:2]- pos[0,:])
    #td_xr[0] =  
    if adj_obs_num == 0:
        loss_gamma = 0.8*c_gamma_p*tanh_function_leader(np.linalg.norm(extra_data[0,0:2]-pos[0,:])-d_quan, d_quan) + 1.0*c_gamma_v*math.sqrt((np.linalg.norm(extra_data[0,2:4]-vel[0,:]))**2)
    else:
        loss_gamma = 0.3*c_gamma_p*tanh_function_leader(np.linalg.norm(extra_data[0,0:2]-pos[0,:])-d_quan, d_quan) + 0.2*c_gamma_v*math.sqrt((np.linalg.norm(extra_data[0,2:4]-vel[0,:]))**2)
    #if loss_gamma > 8:
    #    loss_gamma = 8.0
    #if adj_num==1:
    #    loss_alpha = 80.0
    
    #print('loss_alpha:',loss_alpha)
    return loss_alpha + loss_gamma,vel_err_sum_2 + math.sqrt((np.linalg.norm(extra_data[0,2:4]*2.0))**2)



def dist_weight_function(dit_m,adj):
    dist_matrix= dit_m.copy()
    dist_weight = np.zeros((num_agents,num_agents))
    pro = np.exp(-1.0*(dist_matrix-0.7*d_graph))
    sample_data = -1#random.random()
    pro[pro>=sample_data] = 1.0
    pro[pro<sample_data] = 0
    if np.all(adj==0):
        return np.zeros(num_agents)
    else:
        x_exp = (1.0/np.exp(dist_matrix/1.5))*adj#*pro
        x_sum = np.sum(x_exp, axis = 1, keepdims = True)
        for sum_i in range(num_agents):
            if  x_sum[sum_i]==0:
                dist_weight[sum_i,sum_i]=0.0
            else:
                dist_weight[sum_i] = x_exp[sum_i] / x_sum[sum_i]  
        return dist_weight

def dist_weight_index_function(dit_m,adj): #Communication probability model
    dist_matrix= dit_m.copy()
    #dist_weight = np.zeros((num_agents,num_agents))
    
    #pro = np.exp(-1.0*(dist_matrix-0.7*d_graph))
    
    #pro[pro>=0.6] = 0.6  #最高通信概率0.9
    pro = np.exp(-(dist_matrix**2)/(2*(sigma_c**2)))
    sample_data = 1.0*np.random.random((num_agents,num_agents))
    sample_data = (sample_data + sample_data.T)/2.0 
    pro[pro>=sample_data] = 1.0
    pro[pro<sample_data] = 0

    return pro*adj

def dist_weight_index_function_ideal(dit_m,adj):
    dist_matrix= dit_m.copy()
    #dist_weight = np.zeros((num_agents,num_agents))
    #计算探测概率
    #pro = np.exp(-1.0*(dist_matrix-0.7*d_graph))
    
    #pro[pro>=0.6] = 0.6  #最高通信概率0.9
    pro = np.exp(-(dist_matrix**2)/(2*(sigma_c**2)))
    sample_data = -1 #1.0*np.random.random((num_agents,num_agents))
    # sample_data = (sample_data + sample_data.T)/2.0 
    pro[pro>=sample_data] = 1.0
    pro[pro<sample_data] = 0

    return pro*adj


def dist_weight_index_function_obs(dit_m,adj):
    dist_matrix= dit_m.copy()
    #dist_weight = np.zeros((num_agents,num_agents))
    #计算探测概率
    pro = np.exp(-1.0*(dist_matrix-d_o))
    sample_data = -1#random.random()
    pro[pro>=sample_data] = 1.0
    pro[pro<sample_data] = 0

    return pro*adj

def dist_weight_function_self(dit_m,adj):
    adj_matrix = np.eye(num_agents) + adj
    dist_matrix= dit_m.copy()
    dist_weight = np.zeros((num_agents,num_agents))
    x_exp = (1.0/np.exp(dist_matrix/15000.0))#*adj_matrix
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    for sum_i in range(num_agents):
        if  x_sum[sum_i]==0:
            dist_weight[sum_i,sum_i]=1.0
        else:
            dist_weight[sum_i] = x_exp[sum_i] / x_sum[sum_i]  
    return dist_weight

def dist_weight_function_global(dit_m,adj):
    dist_matrix= dit_m.copy()
    dist_weight = np.zeros((num_agents,num_agents))
    
    x_exp = (1.0/np.exp(dist_matrix/1.5))
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    for sum_i in range(num_agents):
        if  x_sum[sum_i]==0:
            dist_weight[sum_i,sum_i]=1.0
        else:
            dist_weight[sum_i] = x_exp[sum_i] / x_sum[sum_i]  
    return dist_weight

def dist_weight_function_global_obs(dit_m,adj):
    dist_matrix= dit_m.copy()
    dist_weight = np.zeros((1,num_obs))

    #计算探测概率
    pro = np.exp(-1.0*(dist_matrix-d_o))
    sample_data = -1#random.random()
    pro[pro>=sample_data] = 1.0
    pro[pro<sample_data] = 0
    
    x_exp = (1.0/np.exp(dist_matrix/2.0))*adj* pro
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    if x_sum == 0:
        return dist_weight
    else:
        for sum_i in range(num_obs):
            #print((x_exp[0,sum_i] / x_sum[0,sum_i])*adj[0,sum_i]  )
            dist_weight[0,sum_i] = (x_exp[0,sum_i] / x_sum[0])
    return dist_weight

            
def Loss_function_algorithm2_local_obs(dist_agent_obs,agent_obs_nbr):
    dist_obs = dist_agent_obs.copy()
    obs_loss = 0
    for agent_i in range(num_obs):
        tem_d = dist_obs[agent_i]-d_o
        if tem_d >0:
            pass
        else:
            loss_i = agent_obs_nbr[agent_i]*((tem_d)/(0.65*d_o))**4
            obs_loss = obs_loss + loss_i
    return obs_loss*c_obs



def Reward_function(last_loss, current_loss,reward_type): #回报函数
    #print('current_loss: ',current_loss)
    reward_value=0
    if reward_type==0:
        reward_value=0.0-current_loss/100.0
        # if reward_value>0:
        #     print("+++++")
        # if reward_value < -1.0:
        #     reward_value = -1.0
       # if current_loss<0.1:
        #    reward = 0.0
        #    print('---------ok-----------')
        #print('reward_value:',reward_value)
        return  reward_value
    if reward_type==1:
        if current_loss>2:
            reward=(last_loss - current_loss)/last_loss 
            if reward>0:
                reward=reward*0.1
        else:
            reward=0.2*(last_loss - current_loss)/last_loss + (1.0-current_loss/2)*0.8
        if reward>1.0:
            reward=1.0
            print('-----over limit----')
        elif reward<-1.0:
            reward=-1.0
        if current_loss <=0.1:
            reward = 1.0
            print('---------ok-----------')
        return  reward
    

def aciton_select_function(GCN_actor_policy,state,sub_adjacency_matrix,state_obs,sub_adjacency_matrix_obs,agent_i,x_1,v_1,adj,nbr,dist_2,x_r,v_r):#概率决定行为
    global using_flocking_flag
    sample = random.random()
    u_d = np.zeros((1,2))
    
    global epsilon
    global run_step
    run_step=run_step+1
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1* run_step / EPS_DECAY)#+0.0*np.random.dirichlet(0.3*np.ones((1)))
    if run_step%1000==0:
        print('epsilon: ',epsilon)  
    #epsilon=eps_threshold
    #print('epsilon: ',epsilon)
    if sample > -10 :
        using_flocking_flag=False
        with torch.no_grad():
            return_action_mean,return_action_std=GCN_actor_policy(sub_adjacency_matrix,state,sub_adjacency_matrix_obs,state_obs)
            dist = torch.distributions.Normal(return_action_mean, return_action_std)
            return_action = dist.sample()
            return_action=torch.clamp(return_action,-1,1)
            return return_action
    else:
        #return_action=torch.tensor([random.sample(range(2**(2*bits)),1)], device=device, dtype=torch.long)
        #print('return_action_2:',return_action)
        using_flocking_flag=True
        for k in range(1,num_agents):
            u_d=u_d+0.035*nbr[agent_i,k]*adj[agent_i,k]*(np.linalg.norm(x_1[k,:]-x_1[agent_i,:])-d_des)*(x_1[k,:]-x_1[agent_i,:])/np.sqrt(1+dist_2[agent_i,k]) +  0.003*adj[agent_i,k]*(v_1[k,:]-v_1[agent_i,:]) * nbr[agent_i,k]
            #print(0.1*adj[agent_i,k]*(v_1[k,:]-v_1[agent_i,:]) * nbr[agent_i,k])
        u_gamma= -c_gamma_p*Anti_F.limit(x_1[agent_i,:]-x_r,1)-c_gamma_v*(v_1[agent_i,:]-v_r)
            #print( adj[agent_i,k]*(v_1[k,:]-v_1[agent_i,:]) * nbr[agent_i,k])
        return_action = Anti_F.limit(v_1[agent_i,:] + (u_d+u_gamma)*t_gap,1)
        #print('return_action: ',return_action)
        return torch.tensor(return_action, device=device, dtype=torch.double)

def min_elements(array_data,sample_size):

    #find_result=np.where(array_data!=0)
    #if find_result.shape[0]>=sample_size-1:
    min_elements_order = np.argsort(array_data)
    #temp2=min_elements_order[0:sample_size]
    #temp1=np.sort(min_elements_order[0:sample_size])
    return array_data[min_elements_order[0:sample_size]],np.sort(min_elements_order[0:sample_size]),min_elements_order[0:sample_size]
    #else:

    #row_indices,col_indices = row_indices[min_elements_order],col_indices[min_elements_order]
def state_conv_to_3d_tensor(s):
    temp_state=array_tran(s).to(device)

    return temp_state

def state_conv_to_2d_tensor(A):
    temp_A=torch.from_numpy(A).to(device)

    return temp_A

def delay_model(x_1_d,v_1_d,x_1,v_1, adj):
    x_1_new = x_1_d.copy()
    v_1_new = v_1_d.copy()
    for i in range(num_agents):
        if adj[i] == 0:
            x_1_new[i] = x_1[i]
            v_1_new[i] = v_1[i]
    return x_1_new,v_1_new

loss_sys_store=np.zeros((num_episodes,200))
loss_vel_store=np.zeros((num_episodes,200))

if __name__ == '__main__':
    x_r = np.array([[3.0,0]])
    v_r = np.array([[0.5,0.0]])
    sub_graph_dis_norm = np.zeros((sub_graph_size,4))
    sub_graph_vel_norm = np.zeros((sub_graph_size,4))
    sub_graph_dis_norm_local = np.zeros((sub_graph_size,4))
    sub_graph_vel_norm_local = np.zeros((sub_graph_size,4))
    #sub_graph_vel_norm = np.zeros((sub_graph_size,4))
    
    sub_adjacency_nbr = np.zeros((num_agents,num_agents))
    sub_loss_store = np.zeros((num_agents,1))
    sub_loss_obs_store = np.zeros((num_agents,1))
    sub_vel_store = np.zeros((num_agents,1))

    reward = np.zeros((num_agents+1,1))
    eval_reward = 0
    last_eval_reward =-5000
    state_o=np.zeros((num_agents,sub_graph_size+1,4))
    state_o_obs=np.zeros((num_agents,num_obs,4))
    state_o_local = np.zeros((num_agents,sub_graph_size+1,4))
    next_state_o=np.zeros((num_agents,sub_graph_size+1,4))
    next_state_o_obs=np.zeros((num_agents,num_obs,4))
    next_state_o_local=np.zeros((num_agents,sub_graph_size+1,4))
    store_next_state_o = np.zeros((num_agents,sub_graph_size+1,4))
    store_next_adjacency_matrix_global = np.zeros((num_agents+1,num_agents+1))
    store_next_state = np.zeros((num_agents+1,4))
    state=np.zeros((num_agents+num_obs,4))
    next_state=np.zeros((num_agents+num_obs,4))
    store_next_sub_adjacency_matrix = np.zeros((num_agents,sub_graph_size,sub_graph_size))

    current_sub_adjacency_matrix = np.zeros((num_agents,sub_graph_size,sub_graph_size))
    current_sub_adjacency_matrix_loss = np.zeros((num_agents,sub_graph_size,sub_graph_size))
    next_sub_adjacency_matrix_loss = np.zeros((num_agents,sub_graph_size,sub_graph_size))
    current_sub_adjacency_matrix_local = np.zeros((num_agents,sub_graph_size,sub_graph_size))

    next_sub_adjacency_matrix_local = np.zeros((num_agents,sub_graph_size,sub_graph_size))
    dist_weight = np.zeros((num_agents,num_agents,num_agents))
    dist_weight_next = np.zeros((num_agents,num_agents,num_agents))
    dist_weight_self = np.zeros((num_agents,num_agents,num_agents))
    dist_weight_self_next = np.zeros((num_agents,num_agents,num_agents))
    dist_weight_obs = np.zeros((num_agents,num_obs))
    dist_weight_obs_next = np.zeros((num_agents,num_obs))
    adj_flocking = np.zeros((num_agents,num_agents))
    next_sub_adjacency_matrix = np.zeros((num_agents,sub_graph_size,sub_graph_size))
    current_adjacency_matrix_global = np.zeros((num_agents+1,num_agents+1))
    next_adjacency_matrix_global = np.zeros((num_agents+1,num_agents+1))
    temp_current_sub_adjacency_matrix = np.zeros((num_agents,sub_graph_size,sub_graph_size))
    temp_next_sub_adjacency_matrix = np.zeros((num_agents,sub_graph_size,sub_graph_size))

    Loss_weight_matrix_local = np.zeros((num_agents,num_agents))
    Loss_weight_matrix = np.zeros((num_agents,num_agents))
    extra_data = np.zeros((1,4))
    extra_data_pre = np.zeros((1,4))
    #障碍物参数矩阵
    agent_obs_nbr = np.zeros((num_agents,num_obs))
    next_agent_obs_nbr = np.zeros((num_agents,num_obs))

    dist_agent_obs_pre = np.zeros((num_agents,num_obs))
    dist_agent_obs = np.zeros((num_agents,num_obs))
    agent_obs_nbr_loss = np.zeros((num_agents,num_obs))
    average_reward_store = []
    update_rate_flag = False
    for i_episode in range(num_episodes):
        print('------------i_episode----------',i_episode)
        predict_loss = 0
        entropy_value = 0
        logpro_value = 0
        test_loss_all = []
        x_r = np.array([[3.0,0]]) # Virtual Leader Information
        v_r = np.array([[0.6,0.6]])
        des_vel = np.zeros((1,3))
        last_vel = np.zeros((num_agents,time_len,3))
        last_vel_pre = np.zeros((num_agents,num_agents,time_len,3))
        dist_delay = np.zeros((num_agents,num_agents))
        averge_reward=0
        cul_reward=0
        average_reward=0
        counter = 1
        run_time=0.0
       
        current_loss=-100
        next_loss=-100

        #print(np.random.rand(num_agents,2)*10)

        x=(np.random.rand(num_agents,2)-0.5)*10
        #x=np.array([[10.0,0.0],[12.0,13.0],[1.5,6.33],[3.0,16.0],[5,9.33]])
        #x[0,0:2] = x_r
        x_1=x.copy()
        x_temp = np.zeros((num_agents,2))
        #radom action
        v_action=(np.random.rand(num_agents,2)-0.5)*v_max/math.sqrt(2)
        #v_action[0,0:2] = v_r.copy()
        #v_action=np.array([[0.5,0.5],[0.2,0.8],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0],[0.5,0],[0.5,0],[0.5,0]])
        v = v_action.copy()
        v_1 = v.copy()

        last_vel[:,2,0:2] =  v_1[:,0:2].copy()
        for k in range(num_agents):
            last_vel_pre[k,:,] = last_vel.copy()
        #v_norm = Anti_F.action_normalize(v_action,v_max)
        agent_pos_his = np.zeros((num_agents,num_agents,2))
        agent_vel_his = np.zeros((num_agents,num_agents,2))
        obs_pos_his = np.zeros((num_agents,num_obs,2))
        obs_vel_his = np.zeros((num_agents,num_obs,2))
        leader_pos_his = x_r.copy()
        leader_vel_his = v_r.copy()
        # Initialize obstacles
        x_obs = np.zeros((num_obs,2))
        # x_obs[:,0:1] = (np.random.rand(num_obs,1)-0.5)*20 
        # x_obs[:,1:2] = (np.random.rand(num_obs,1)-0.5)*10 
        # x_obs = x_obs + np.array([[50,50.0]])
        # if i_episode%5 == 0:
        #     x_obs = np.array([[30,35.0],[40,33.0],[38,46.0]])
        # else:
        #     x_obs = np.array([[130,135.0],[140,133.0],[138,146.0]])
        x_obs = np.array([[30,35.0],[45,33.0],[38,56.0]])
        v_obs = (np.random.rand(num_obs,2)-0.5)*0.2#np.zeros((num_obs,2))
        x_obs_1 = x_obs.copy()
        v_obs_1 = v_obs.copy()
        #v_obs[:,0:1] = -np.ones((num_obs,1)) *0.5
        #v_obs[:,1:2] = np.random.randn(num_obs,1)*0.1
        #Initialize the previous motion state
        for k in range(num_agents):
            agent_pos_his[k] = x.copy()
            agent_vel_his[k] = v.copy()
            obs_pos_his[k] = x_obs.copy()
            obs_vel_his[k] = v_obs.copy()




        #Calculate adjacency matrix
        dist_gap = Anti_F.get_gap(x_1)
        dist_2 = Anti_F.squd_norm(dist_gap)
        dist = np.sqrt(dist_2) 
        adj_flocking,dist_nbr = Anti_F.flock_func(dist,d_des,0.5)  
        nbr = np.zeros((num_agents,num_agents))
        nbr_temp = np.zeros((num_agents,num_agents))
        sub_adjacency_nbr = np.zeros((num_agents,num_agents))
        nbr_temp[dist<=d_graph] = 1  
        sub_adjacency_nbr[dist<=0.7*d_graph]=1
        adjacency_matrix=sub_adjacency_nbr - np.identity(num_agents) #<8.4
        adjacency_matrix_pre = adjacency_matrix.copy()

        index_matrix_local = np.zeros((num_agents,num_agents))
        index_matrix_local = adjacency_matrix.copy()
        nbr = nbr_temp - np.identity(num_agents)   #<12
        #nbr_pre = nbr.copy()
        adj_d=adjacency_matrix*dist #

        
        
        #Calculate initial  state
        for i_one_hot in range(0,num_agents):
            #action_temp=Anti_F.action_encode(v_norm[i_one_hot],encode_bit)
            temp_dis_i = dist[i_one_hot]
            min_value,min_index, min_index_mix = min_elements(temp_dis_i,sub_graph_size) #返回最小的7个值及其位置， min_index排序后的位置，min_index_mix排序前位置
            #sub_adjacency_matrix[i_one_hot,:,:]=adjacency_matrix[min_index,min_index]#7阶子图邻接矩阵
            adj_num = np.sum(nbr_temp[i_one_hot])    
            adj_num_loss = np.sum(sub_adjacency_nbr[i_one_hot])
             #Current Agent Observations
            extra_data[0,0:2] = Anti_F.limit_state((x_r-x_1[i_one_hot])/d_graph,1.5)
            extra_data[0,2:4] = Anti_F.limit_state(( v_r-v_1[i_one_hot] )/2.0,1.0)
            

            sub_graph_dis_norm = Anti_F.distance_normalize(x_1[min_index_mix],min_index_mix,0,d_graph)
            sub_graph_vel_norm = Anti_F.co_vel_normalize_1(v_1[min_index_mix],min_index_mix,0,1.0)
            sub_graph_dis_norm_local = sub_graph_dis_norm.copy()
            sub_graph_vel_norm_local = sub_graph_vel_norm.copy()
            adjacency_graph_matrix = adjacency_matrix_sample(num_agents,min_index_mix)
            current_sub_adjacency_matrix[i_one_hot,:,:] = np.dot(np.dot(adjacency_graph_matrix.T, nbr),adjacency_graph_matrix) #通信邻接矩阵
            current_sub_adjacency_matrix_loss[i_one_hot,:,:] = np.dot(np.dot(adjacency_graph_matrix.T, adjacency_matrix),adjacency_graph_matrix) #邻接矩阵 
            dist_sub = np.dot(np.dot(adjacency_graph_matrix.T, dist),adjacency_graph_matrix) #距离矩阵
            
            dist_weight_index = dist_weight_index_function_ideal(dist_sub,current_sub_adjacency_matrix[i_one_hot,:,:]) 
            dist_weight[i_one_hot] = dist_weight_function(dist_sub,dist_weight_index )  #第一行有用
            dist_weight_self[i_one_hot] = dist_weight_function_self(dist_sub,current_sub_adjacency_matrix[i_one_hot,:,:])  #第一行有用
            
            current_sub_adjacency_matrix_local[i_one_hot,:,:] = current_sub_adjacency_matrix[i_one_hot,:,:].copy()
            #Loss_weight_matrix = Loss_weight(sub_graph_dis_norm,sub_graph_vel_norm,d_des,d_graph,current_sub_adjacency_matrix[i_one_hot,:,:])/10.0
            #Loss_weight_matrix_local = Loss_weight_matrix.copy()
        
            if adj_num<num_agents:
                sub_graph_dis_norm_local[int(adj_num):int(num_agents)] = 0
                sub_graph_vel_norm_local[int(adj_num):int(num_agents)] = 0
                current_sub_adjacency_matrix_local[i_one_hot,int(adj_num):int(num_agents),:] = 0   ##
                current_sub_adjacency_matrix_local[i_one_hot,:,int(adj_num):int(num_agents)] = 0   ###

            # if i_one_hot == 0:
            #     sub_loss_store[i_one_hot] = Loss_function_algorithm2_local_leader(sub_graph_dis_norm,sub_graph_vel_norm,x_r,v_r,d_des,d_graph)
                
            # else:
            #     #adj_num = np.sum(sub_adjacency_nbr[i_one_hot])
            #sub_loss_store[i_one_hot] = Loss_function_algorithm2_local(sub_graph_dis_norm,sub_graph_vel_norm,min_index_mix,d_des,d_graph,adj_num,current_sub_adjacency_matrix[i_one_hot,:,:],extra_data)

               

                   
            #current_sub_adjacency_matrix_local[i_one_hot,:,:] = Loss_weight_matrix_local.copy()
            #current_sub_adjacency_matrix[i_one_hot,:,:] =   normalization_deg(current_sub_adjacency_matrix[i_one_hot,:,:])#Loss_weight_matrix.copy()
            #current_sub_adjacency_matrix_local[i_one_hot,:,:] =   normalization_deg(current_sub_adjacency_matrix_local[i_one_hot,:,:])#Loss_weight_matrix.copy()
            

            state_o_p_v_local=np.hstack((sub_graph_dis_norm_local,sub_graph_vel_norm_local))
            state_o_p_v_local[dist_weight_index[0,:]== 0] = 0.0
            
            state_o_local[i_one_hot] = np.vstack((state_o_p_v_local,extra_data))

            state_o_p_v=np.hstack((sub_graph_dis_norm,sub_graph_vel_norm))
            state_o[i_one_hot] = state_o_local[i_one_hot].copy()#np.vstack((state_o_p_v,extra_data))#np.hstack((state_o_p_v,F_L_info))

            #Calculate external observation states

            agent_obs_nbr = np.zeros((num_agents,num_obs))  
            agent_obs_nbr_loss = np.zeros((num_agents,num_obs))
            x_obs_gap = x_obs - x_1[i_one_hot,:]
            dist_agent_obs[i_one_hot,:] = np.sqrt(x_obs_gap[:,0]**2 + x_obs_gap[:,1]**2)
            agent_obs_nbr[i_one_hot,dist_agent_obs[i_one_hot,:]<r_o] = 1.0
            agent_obs_nbr_loss[i_one_hot,dist_agent_obs[i_one_hot,:]<d_o] = 1.0
            adj_obs_num = np.sum(agent_obs_nbr_loss[i_one_hot])
            
            #Rearrange obstacle information based on distance
            temp_obs_dist_i = dist_agent_obs[i_one_hot]
            min_value_obs,min_index_obs, min_index_mix_obs = min_elements(temp_obs_dist_i,num_obs)

            x_obs_norm_nbr =  ((x_obs - x_1[i_one_hot])/r_o)*(agent_obs_nbr[i_one_hot:i_one_hot+1,:]).transpose()
            x_obs_nbr_order = x_obs_norm_nbr[min_index_mix_obs]

            v_obs_norm_nbr =  ((v_obs - v_1[i_one_hot])/2.0)*(agent_obs_nbr[i_one_hot:i_one_hot+1,:]).transpose()
            v_obs_nbr_order = v_obs_norm_nbr[min_index_mix_obs]

            state_o_obs_p_v = np.hstack((x_obs_nbr_order ,v_obs_nbr_order))

            # x_obs_gap = x_1[i_one_hot,:] - x_obs[min_index_mix_obs]
            # dist_agent_obs[i_one_hot,:] = np.sqrt(x_obs_gap[:,0]**2 + x_obs_gap[:,1]**2)
            # agent_obs_nbr[i_one_hot,dist_agent_obs[i_one_hot,:]<r_o] = 1.0
            dist_agent_obs[i_one_hot,:] = dist_agent_obs[i_one_hot,min_index_mix_obs]
            agent_obs_nbr[i_one_hot,:] = agent_obs_nbr[i_one_hot,min_index_mix_obs]

            dist_weight_obs[i_one_hot:i_one_hot+1] = dist_weight_function_global_obs(dist_agent_obs[i_one_hot:i_one_hot+1,:],agent_obs_nbr[i_one_hot:i_one_hot+1,:])
            dist_weight_index_obs = dist_weight_index_function_obs(dist_agent_obs[i_one_hot:i_one_hot+1,:],agent_obs_nbr[i_one_hot:i_one_hot+1,:]) 
            state_o_obs_p_v[dist_weight_index_obs[0]==0] = 0.0
            state_o_obs[i_one_hot] = state_o_obs_p_v.copy()

            sub_loss_obs_store[i_one_hot] = Loss_function_algorithm2_local_obs(dist_agent_obs[i_one_hot,:],agent_obs_nbr[i_one_hot,:])
            
            sub_loss_store[i_one_hot],sub_vel_store[i_one_hot] = Loss_function_algorithm2_local(sub_graph_dis_norm,sub_graph_vel_norm,min_index_mix,d_des,d_graph,adj_num_loss,current_sub_adjacency_matrix_loss[i_one_hot,:,:],extra_data,adj_obs_num,v_1)

        
        
        
        current_loss = np.sum(sub_loss_store)+np.sum(sub_loss_obs_store)
        cuurent_vel_loss = np.sum(sub_vel_store)/42.0
        # loss_sys_store[i_episode,0]=-current_loss/100
        # loss_vel_store[i_episode,0]= cuurent_vel_loss
        

        

        while(True):
            
            #环境交互
            
            x_r = x_r + v_r * t_gap
            x_obs = x_obs_1 + v_obs*t_gap
            #print('vel:',v_1)
            x_temp = x_1.copy()
            for i_one_hot in range(num_agents):
                #Getting actions
                action[i_one_hot]= aciton_select_function(GCN_actor_policy,state_conv_to_3d_tensor(state_o[i_one_hot]),state_conv_to_3d_tensor(dist_weight[i_one_hot,:,:]),state_conv_to_3d_tensor(state_o_obs[i_one_hot]),state_conv_to_3d_tensor(dist_weight_obs[i_one_hot:i_one_hot+1,:]),i_one_hot,x_1,v_1,adj_flocking,nbr,dist_2,x_r,v_r)
                temp_action=action[i_one_hot].to("cpu")
                numpy_action=temp_action.detach().numpy()
                v_action[i_one_hot] = numpy_action
                v_delta = numpy_action.copy()
                
                if not using_flocking_flag: #or epsilon<1:
                    v_action[i_one_hot,0] = np.clip(np.random.normal(v_action[i_one_hot,0], var), -0.99, 0.99)
                    v_action[i_one_hot,1] = np.clip(np.random.normal(v_action[i_one_hot,1], var), -0.99, 0.99)

                # if i_one_hot == 0: #leader 运动
                #     v_action[i_one_hot,0:2] = np.zeros((1,2))
                #     v[i_one_hot,0:2] = v_r + v_action[i_one_hot,0:2]
                #     x[i_one_hot,0:2] = x_r
                # else:

                des_vel[0,0:2] = Anti_F.limit_v(v_delta[0]+ v_1[i_one_hot,:],1.5)
                #last_vel[0,0:2] = v_1[i_one_hot,0:2].copy()

                    #des_vel_input = last_vel + (des_vel- last_vel) * (0.2*(j+1))
                des_vel_input = des_vel.copy()
                acture_vel = 1.5*motion_model(state_conv_to_2d_tensor(last_vel[i_one_hot:i_one_hot+1])/1.5,state_conv_to_2d_tensor(des_vel_input)/1.5).to('cpu')
                # print("acture_vel of ",i_one_hot,": ",acture_vel)
                # print("des_vel_input",des_vel_input)
                #print('GT:',last_vel[i_one_hot:i_one_hot+1],i_one_hot)
                v[i_one_hot,0:2] = (acture_vel.detach().numpy())[0,0:2]#Anti_F.limit_v(v_action[i_one_hot,:]+ v_1[i_one_hot,:],1.5)#
                #更新last_vel
                last_vel[i_one_hot,0:2,0:2] = last_vel[i_one_hot,1:3,0:2].copy()
                last_vel[i_one_hot,2,0:2] = v[i_one_hot,0:2].copy()
                x[i_one_hot,0:2] = x_1[i_one_hot,0:2] + v[i_one_hot,0:2]*t_gap + np.random.randn(1,2)*0.0

            #print('v_action:',v_action)    
            v_action_store = v_action.copy()
            #print(v)
            if  counter%40==0:
                    print('vel:',v)
            dist_gap = Anti_F.get_gap(x)
            dist_2 = Anti_F.squd_norm(dist_gap)
            dist=np.sqrt(dist_2)

            adj_flocking,dist_nbr = Anti_F.flock_func(dist,d_des,0.5)   
            nbr = np.zeros((num_agents,num_agents))
            nbr_temp = np.zeros((num_agents,num_agents))
            sub_adjacency_nbr = np.zeros((num_agents,num_agents))
            #nbr[dist<=r_c]=1 
            nbr_temp[dist<=d_graph]=1   
            sub_adjacency_nbr[dist<=0.7*d_graph]=1
            #dist_delay = delay_model(sub_adjacency_nbr)
            #sub_adjacency_nbr[0,:]=1
            #sub_adjacency_nbr[:,0]=1
            adjacency_matrix=sub_adjacency_nbr - np.identity(num_agents)
            
            index_matrix_local = np.zeros((num_agents,num_agents))
            index_matrix_local[dist<=d_graph]=1
            nbr = nbr_temp - np.identity(num_agents)
            adj_d=adjacency_matrix*dist #d_graph内的邻居     
                
            #Calculate new status
            
            for i_one_hot in range(num_agents):
                x_1_d = x.copy()
                v_1_d = v.copy()
                # x_1_d,v_1_d = delay_model(x_1_d,v_1_d,x_1,v_1,index_matrix_local[i_one_hot])
                # Random communication model
                
                next_dist_communicate_index = dist_weight_index_function(dist,nbr)  #随机通信模型
                next_dist_communicate_index_data = next_dist_communicate_index + np.identity(num_agents)
               # print(next_dist_communicate_index)
                x_1_d[next_dist_communicate_index_data[i_one_hot,:]== 0] = -70
                v_1_d[next_dist_communicate_index_data[i_one_hot,:]== 0] = 0

                # Behavior Reasoning Model
                for k_i in range(num_agents):
                    if adjacency_matrix_pre[i_one_hot,k_i] == 1 and x_1_d[k_i,0] == -70 and agent_pos_his[i_one_hot,k_i,0] != 0: #满足预测条件:不建立通信，但上一时刻在通信范围内，并且上一时刻已交互信息
                        # Calculate predicted observation status
                        x_1_d_pre = agent_pos_his[i_one_hot]
                        v_1_d_pre = agent_vel_his[i_one_hot]
                        dist_gap_pre = Anti_F.get_gap(x_1_d_pre) # 利用上一时刻观测计算距离矩阵
                        dist_2_pre = Anti_F.squd_norm(dist_gap_pre)
                        dist_pre=np.sqrt(dist_2_pre)
                        nbr_pre = np.zeros((num_agents,num_agents))
                        nbr_pre[dist_pre<=d_graph]=1  
                        nbr_pre = nbr_pre - np.identity(num_agents)
                        adj_num_pre = np.sum(nbr_pre[k_i]) + 1  

                        temp_dis_i_pre=dist_pre[k_i]
                        min_value_pre,min_index_pre, min_index_mix_pre=min_elements(temp_dis_i_pre,sub_graph_size)
                        extra_data_pre[0,0:2] = Anti_F.limit_state((leader_pos_his - x_1_d_pre[k_i] )/d_graph,1.5)
                        extra_data_pre[0,2:4] = Anti_F.limit_state((leader_vel_his - v_1_d_pre[k_i])/2.0,1.0)
                        sub_graph_dis_norm_pre = Anti_F.distance_normalize(x_1_d_pre[min_index_mix_pre],min_index_mix_pre,0,d_graph)
                        sub_graph_vel_norm_pre = Anti_F.co_vel_normalize_1(v_1_d_pre[min_index_mix_pre],min_index_mix_pre,0,1.0)   #相对运动信息

                        adjacency_graph_matrix_local_pre = adjacency_matrix_sample(num_agents,min_index_mix_pre)
                        next_sub_adjacency_matrix_pre = np.dot(np.dot(adjacency_graph_matrix_local_pre.T, nbr_pre),adjacency_graph_matrix_local_pre) 
                        
                        dist_sub_pre = np.dot(np.dot(adjacency_graph_matrix_local_pre.T, dist_pre),adjacency_graph_matrix_local_pre) 
                        dist_weight_pre = dist_weight_function(dist_sub_pre,next_sub_adjacency_matrix_pre)   #智能体之间注意力权重
                        if adj_num_pre<num_agents:
                            sub_graph_dis_norm_pre[int(adj_num_pre):int(num_agents)] = 0
                            sub_graph_vel_norm_pre[int(adj_num_pre):int(num_agents)] = 0
                            

                        state_o_p_v_pre = np.hstack((sub_graph_dis_norm_pre,sub_graph_vel_norm_pre))
                        state_o_pre = np.vstack((state_o_p_v_pre,extra_data_pre))
                        #Calculate obstacle observation states
                        next_agent_obs_nbr_pre = np.zeros((num_agents,num_obs))       
                        x_obs_pre = obs_pos_his[i_one_hot]
                        v_obs_pre = obs_vel_his[i_one_hot]
                        x_obs_gap_pre  = x_obs_pre - x_1_d_pre[k_i,:]
                        dist_agent_obs_pre[k_i,:] = np.sqrt(x_obs_gap_pre[:,0]**2 + x_obs_gap_pre[:,1]**2)
                        next_agent_obs_nbr_pre[dist_agent_obs_pre<r_o] = 1.0
                        
                        #mask
                        next_agent_obs_nbr_current_pre = np.zeros((num_agents,num_obs))
                        x_obs_gap_pre_current  = x_obs_pre - x_1_d_pre[i_one_hot,:]
                        dist_agent_obs_pre_current = np.sqrt(x_obs_gap_pre_current [:,0]**2 + x_obs_gap_pre_current [:,1]**2)
                        next_agent_obs_nbr_current_pre[k_i,dist_agent_obs_pre_current<r_o] = 1.0
                        next_agent_obs_nbr_pre[k_i] = next_agent_obs_nbr_pre[k_i] * next_agent_obs_nbr_current_pre[k_i]
                        dist_agent_obs_pre[k_i,next_agent_obs_nbr_current_pre[k_i]==0] = 50
                        #重新排列障碍物信息
                        temp_obs_dist_i_pre = dist_agent_obs_pre[k_i]
                        min_value_obs_pre,min_index_obs_pre, min_index_mix_obs_pre = min_elements(temp_obs_dist_i_pre,num_obs)

                        x_obs_norm_nbr_pre =  ((x_obs_pre - x_1_d_pre[k_i])/r_o)*(next_agent_obs_nbr_pre[k_i:k_i+1,:]).transpose()
                        x_obs_nbr_order_pre = x_obs_norm_nbr_pre[min_index_mix_obs_pre]

                        v_obs_norm_nbr_pre =  ((v_obs_pre - v_1_d_pre[k_i])/2.0)*(next_agent_obs_nbr_pre[k_i:k_i+1,:]).transpose()
                        v_obs_nbr_order_pre = v_obs_norm_nbr_pre[min_index_mix_obs_pre]
                        state_o_obs_pre = np.hstack((x_obs_nbr_order_pre ,v_obs_nbr_order_pre))
                        #Obstacle distance weight
                        dist_agent_obs_pre[k_i]= dist_agent_obs_pre[k_i,min_index_mix_obs]
                        next_agent_obs_nbr_pre[k_i] = next_agent_obs_nbr_pre[k_i,min_index_mix_obs]
                        dist_weight_obs_pre = dist_weight_function_global_obs(dist_agent_obs_pre[k_i:k_i+1,:],next_agent_obs_nbr_pre[k_i:k_i+1,:])

                        store_state_o_pre = state_o_pre.copy()
                        store_dist_weight_pre = dist_weight_pre.copy()
                        store_state_o_obs_pre = state_o_obs_pre.copy()
                        store_dist_weight_obs_pre = dist_weight_obs_pre.copy()
                        store_action_pre = v_action[k_i].copy()
                        test_action_pre = v_action[k_i:k_i+1].copy()
                        # Storage sample 
                        # namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_obs','action'))
                        Agent_Mem_predicate.push(state_conv_to_2d_tensor(store_state_o_pre), state_conv_to_2d_tensor(store_state_o_obs_pre), state_conv_to_2d_tensor(store_dist_weight_pre), state_conv_to_2d_tensor(store_dist_weight_obs_pre),state_conv_to_2d_tensor(store_action_pre))
                        # Motion information prediction
                        action_pre = aciton_select_function(GCN_actor_predicate,state_conv_to_3d_tensor(state_o_pre ),state_conv_to_3d_tensor(dist_weight_pre),state_conv_to_3d_tensor(state_o_obs_pre),state_conv_to_3d_tensor(dist_weight_obs_pre),k_i,x_1,v_1,adj_flocking,nbr,dist_2,x_r,v_r)
                        predicate_error = F.mse_loss(action_pre, state_conv_to_2d_tensor(test_action_pre))
                        test_loss_all.append(predicate_error.detach().item()) 
                        #print("action predicate:",action_pre)
                        #action_true = aciton_select_function(GCN_actor_policy,state_conv_to_3d_tensor(state_o[k_i]),state_conv_to_3d_tensor(dist_weight[k_i,:,:]),state_conv_to_3d_tensor(state_o_obs[k_i]),state_conv_to_3d_tensor(dist_weight_obs[k_i:k_i+1,:]),k_i,x_1,v_1,adj_flocking,nbr,dist_2,x_r,v_r)
                        #print("action GT:",action_true)
                        #print("action GT_2:",store_action_pre)
                        temp_action_pre = action_pre.to("cpu")
                        v_action_pre = temp_action_pre.detach().numpy()

                        des_vel[0,0:2] = Anti_F.limit_v(v_action_pre+ v_1_d_pre[k_i,:],1.5)
                        #last_vel_pre[:,0:2] = v_1_d_pre.copy()

                            #des_vel_input = last_vel + (des_vel- last_vel) * (0.2*(j+1))
                        des_vel_input_pre = des_vel.copy()
                        #print('pre:',last_vel_pre[i_one_hot,k_i:k_i+1],k_i)
                        
                        acture_vel_pre = 1.5*motion_model(state_conv_to_2d_tensor(last_vel_pre[i_one_hot,k_i:k_i+1])/1.5,state_conv_to_2d_tensor(des_vel_input_pre)/1.5).to('cpu')
                        
                        # print("predicate acture_vel of ",k_i,": ",acture_vel_pre)
                        # print("des_vel_input",des_vel_input)
                        # Complete observation information
                        v_1_d[k_i,0:2] = (acture_vel_pre.detach().numpy())[0,0:2]#Anti_F.limit_v(v_action[i_one_hot,:]+ v_1[i_one_hot,:],1.5)#
                        #print('predict_error: ', np.linalg.norm(v_1_d[k_i,0:2]- v[k_i,0:2]),"distance: ",np.linalg.norm(x[k_i,0:2]- x[i_one_hot,0:2]))
                        x_1_d[k_i,0:2] = x_1_d_pre[k_i,0:2] + v_1_d[k_i,0:2]*t_gap 

                        # if np.sum(next_agent_obs_nbr_pre[k_i])>0:
                        #     print('wait')



                agent_pos_his[i_one_hot] = x_1_d.copy()
                agent_vel_his[i_one_hot] = v_1_d.copy()
                
                last_vel_pre[i_one_hot,:,0:2,0:2] = last_vel_pre[i_one_hot,:,1:3,0:2].copy()
                last_vel_pre[i_one_hot,:,2,0:2] = v_1_d.copy()

                #print('his_pre:',last_vel_pre[i_one_hot])
                #print('his:',last_vel[i_one_hot])
                
                obs_pos_his[i_one_hot] = x_obs.copy()
                obs_vel_his[i_one_hot] = v_obs.copy()

                # 重新计算邻接矩阵 
                dist_gap_re = Anti_F.get_gap(x_1_d)
                dist_2_re = Anti_F.squd_norm(dist_gap_re)
                dist_re=np.sqrt(dist_2_re)
                nbr_re = np.zeros((num_agents,num_agents))
                nbr_re_temp = np.zeros((num_agents,num_agents))
                nbr_re_temp[dist_re<=d_graph]=1  
                nbr_re = nbr_re_temp - np.identity(num_agents)


                temp_dis_i=dist_re[i_one_hot]
                min_value,min_index, min_index_mix=min_elements(temp_dis_i,sub_graph_size)
                min_value_loss,min_index_loss, min_index_mix_loss=min_elements(dist[i_one_hot],sub_graph_size)
                adj_num = np.sum(nbr_re_temp[i_one_hot])
                adj_num_loss = np.sum(sub_adjacency_nbr[i_one_hot])
                extra_data[0,0:2] = Anti_F.limit_state((x_r - x[i_one_hot] )/d_graph,1.5)
                extra_data[0,2:4] = Anti_F.limit_state((v_r - v[i_one_hot])/2.0,1.0)
                
                
                sub_graph_dis_norm = Anti_F.distance_normalize(x_1_d[min_index_mix],min_index_mix,0,d_graph)
                sub_graph_vel_norm = Anti_F.co_vel_normalize_1(v_1_d[min_index_mix],min_index_mix,0,1.0)
                sub_graph_dis_norm_reward = Anti_F.distance_normalize(x[min_index_mix_loss],min_index_mix_loss,0,d_graph)
                sub_graph_vel_norm_reward = Anti_F.co_vel_normalize_1(v[min_index_mix_loss],min_index_mix_loss,0,1.0)

                sub_graph_dis_norm_local = sub_graph_dis_norm.copy()
                sub_graph_vel_norm_local = sub_graph_vel_norm.copy()
                adjacency_graph_matrix_local = adjacency_matrix_sample(num_agents,min_index_mix)
                adjacency_graph_matrix_local_loss = adjacency_matrix_sample(num_agents,min_index_mix_loss)
                next_sub_adjacency_matrix[i_one_hot,:,:] = np.dot(np.dot(adjacency_graph_matrix_local.T, nbr_re),adjacency_graph_matrix_local) 
                next_sub_adjacency_matrix_loss[i_one_hot,:,:] = np.dot(np.dot(adjacency_graph_matrix_local_loss.T, adjacency_matrix),adjacency_graph_matrix_local_loss) 
                next_sub_adjacency_matrix_local[i_one_hot,:,:] = next_sub_adjacency_matrix[i_one_hot,:,:].copy()
                dist_sub = np.dot(np.dot(adjacency_graph_matrix_local.T, dist_re),adjacency_graph_matrix_local) 
                
                #next_dist_weight_index = np.dot(np.dot(adjacency_graph_matrix_local.T, next_dist_communicate_index ),adjacency_graph_matrix_local)
                # next_dist_weight_index = dist_weight_index_function(dist_sub,next_sub_adjacency_matrix[i_one_hot,:,:]) 
                dist_weight_next[i_one_hot] = dist_weight_function(dist_sub,next_sub_adjacency_matrix[i_one_hot,:,:])
                dist_weight_self_next[i_one_hot] = dist_weight_function_self(dist_sub,next_sub_adjacency_matrix[i_one_hot,:,:])
                
                #Loss_weight_matrix = Loss_weight(sub_graph_dis_norm,sub_graph_vel_norm,d_des,d_graph,next_sub_adjacency_matrix[i_one_hot,:,:])/10.0
                #Loss_weight_matrix_local = Loss_weight_matrix.copy()
                if adj_num<num_agents:
                    sub_graph_dis_norm_local[int(adj_num):int(num_agents)] = 0.0
                    sub_graph_vel_norm_local[int(adj_num):int(num_agents)] = 0
                    #next_F_L_info_local[int(adj_num):int(num_agents)] = 0
                    next_sub_adjacency_matrix_local[i_one_hot,int(adj_num):int(num_agents)] = 0
                    next_sub_adjacency_matrix_local[i_one_hot,:,int(adj_num):int(num_agents)] = 0

                
                
                # if i_one_hot == 0:
                #     sub_loss_store[i_one_hot] = Loss_function_algorithm2_local_leader(sub_graph_dis_norm,sub_graph_vel_norm,d_des,d_graph)
                    
                # else:
                #     #adj_num = np.sum(sub_adjacency_nbr[i_one_hot])
        
                #sub_loss_store[i_one_hot] = Loss_function_algorithm2_local(sub_graph_dis_norm_reward,sub_graph_vel_norm_reward,min_index_mix,d_des,d_graph,adj_num,next_sub_adjacency_matrix[i_one_hot,:,:],extra_data)
                    
            
                #next_sub_adjacency_matrix_local[i_one_hot,:,:] = normalization_deg(next_sub_adjacency_matrix_local[i_one_hot,:,:])
                #next_sub_adjacency_matrix[i_one_hot,:,:] = normalization_deg(next_sub_adjacency_matrix[i_one_hot,:,:])#Loss_weight_matrix.copy()

                next_state_o_p_v_local=np.hstack((sub_graph_dis_norm_local,sub_graph_vel_norm_local))
                #next_state_o_p_v_local[next_dist_communicate_index_pre[0,:]== 0] = 0.0
                next_state_o_local[i_one_hot] = np.vstack((next_state_o_p_v_local,extra_data))

                next_state_o_p_v=np.hstack((sub_graph_dis_norm,sub_graph_vel_norm))
                next_state_o[i_one_hot] = next_state_o_local[i_one_hot].copy()#np.vstack((next_state_o_p_v,extra_data))
                #sub_adjacency_matrix[i_one_hot,:,:]=adjacency_matrix[min_index,min_index]#7阶子图邻接矩阵
               
               #next_sub_adjacency_matrix[i_one_hot,:,:] = np.identity(sub_graph_size)#(next_sub_adjacency_matrix[i_one_hot,:,:] * (temp_dis_next_sub_adjacency_matrix[next_sub_graph_index[0],:]).T)
                #print(sub_adjacency_matrix[i_one_hot,:,:])
                
                #计算与障碍物邻接矩阵
                next_agent_obs_nbr = np.zeros((num_agents,num_obs))  
                agent_obs_nbr_loss = np.zeros((num_agents,num_obs))
                x_obs_gap = x_obs - x[i_one_hot,:]
                dist_agent_obs[i_one_hot,:] = np.sqrt(x_obs_gap[:,0]**2 + x_obs_gap[:,1]**2)
                next_agent_obs_nbr[dist_agent_obs<r_o] = 1.0
                agent_obs_nbr_loss[i_one_hot,dist_agent_obs[i_one_hot,:]<d_o] = 1.0
                adj_obs_num = np.sum(agent_obs_nbr_loss[i_one_hot])
                #重新排列障碍物信息
                temp_obs_dist_i = dist_agent_obs[i_one_hot]
                min_value_obs,min_index_obs, min_index_mix_obs = min_elements(temp_obs_dist_i,num_obs)

                x_obs_norm_nbr =  ((x_obs - x[i_one_hot])/r_o)*(next_agent_obs_nbr[i_one_hot:i_one_hot+1,:]).transpose()
                x_obs_nbr_order = x_obs_norm_nbr[min_index_mix_obs]

                v_obs_norm_nbr =  ((v_obs - v[i_one_hot])/2.0)*(next_agent_obs_nbr[i_one_hot:i_one_hot+1,:]).transpose()
                v_obs_nbr_order = v_obs_norm_nbr[min_index_mix_obs]
                next_state_o_obs_pv = np.hstack((x_obs_nbr_order ,v_obs_nbr_order))

                dist_agent_obs[i_one_hot,:] = dist_agent_obs[i_one_hot,min_index_mix_obs]
                next_agent_obs_nbr[i_one_hot,:] = next_agent_obs_nbr[i_one_hot,min_index_mix_obs]
                dist_weight_obs_next[i_one_hot:i_one_hot+1] = dist_weight_function_global_obs(dist_agent_obs[i_one_hot:i_one_hot+1,:],next_agent_obs_nbr[i_one_hot:i_one_hot+1,:])
                next_dist_weight_index_obs = dist_weight_index_function_obs(dist_agent_obs[i_one_hot:i_one_hot+1,:],next_agent_obs_nbr[i_one_hot:i_one_hot+1,:])
                next_state_o_obs_pv[next_dist_weight_index_obs[0]==0] = 0.0
                next_state_o_obs[i_one_hot,:] = next_state_o_obs_pv.copy()
                # Calculating Rewards
                sub_loss_store[i_one_hot],sub_vel_store[i_one_hot] = Loss_function_algorithm2_local(sub_graph_dis_norm_reward,sub_graph_vel_norm_reward,min_index_mix_loss,d_des,d_graph,adj_num_loss,next_sub_adjacency_matrix_loss[i_one_hot,:,:],extra_data,adj_obs_num,v)
                sub_loss_obs_store[i_one_hot] = Loss_function_algorithm2_local_obs(dist_agent_obs[i_one_hot,:],next_agent_obs_nbr[i_one_hot,:])
                #next_state_o[i_one_hot]=np.hstack((sub_graph_dis_norm,sub_graph_vel_norm))
                  
            leader_pos_his = x_r.copy()
            leader_vel_his = v_r.copy()
            adjacency_matrix_pre = adjacency_matrix.copy()
            store_next_state_o = next_state_o.copy()
            store_next_sub_adjacency_matrix = next_sub_adjacency_matrix.copy()
            #print(next_state_o[0,:,:])
            #print('index:',min_index_mix_all)
            #sub_loss_store = sub_loss_store[store_min_index_mix_all[1:num_agents+1]]
            #sub_loss_obs_store = sub_loss_obs_store[store_min_index_mix_all[1:num_agents+1]]
            #print('store index:',store_min_index_mix_all)

            #Calculate cumulative reward
            next_loss_agent = np.sum(sub_loss_store)
            next_loss_obs = np.sum(sub_loss_obs_store)
            next_loss = next_loss_agent + next_loss_obs
            reward[0]=Reward_function(current_loss,next_loss,0)
            next_vel_loss = np.sum(sub_vel_store)/42.0
            
            # if counter<= 199:
            #    loss_sys_store[i_episode,counter]=reward[0]
            #    loss_vel_store[i_episode,counter]=next_vel_loss

            for i_reward in range(1,num_agents+1):
                reward[i_reward] = Reward_function(sub_loss_store[i_reward-1]+sub_loss_obs_store[i_reward-1], sub_loss_store[i_reward-1]+sub_loss_obs_store[i_reward-1],0)*num_agents/2.0

            
            cul_reward=cul_reward + reward[0]
            current_loss = next_loss


            store_dist_weight_self_next = dist_weight_self_next.copy()
            sore_dist_weight_next = dist_weight_next.copy()
            store_dist_weight_obs_next =dist_weight_obs_next.copy()
            store_next_state_o_obs = next_state_o_obs.copy()

            #Push into experience replay buffer
                #if len(Agent_Mem)<=capacity_size:
            #if reward > 0.0:dist_weight_next
            #Transition = namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_obs','action', 'next_state_o','next_state_o_obs','next_dist_weight','next_dist_weight_obs', 'reward'))
            Agent_Mem_H.push(state_conv_to_2d_tensor(state_o),state_conv_to_2d_tensor(state_o_obs),state_conv_to_2d_tensor(dist_weight),state_conv_to_2d_tensor(dist_weight_self),state_conv_to_2d_tensor(dist_weight_obs),\
                        state_conv_to_3d_tensor(v_action_store),state_conv_to_2d_tensor(store_next_state_o),state_conv_to_2d_tensor(store_next_state_o_obs),\
                        state_conv_to_2d_tensor(sore_dist_weight_next),state_conv_to_2d_tensor(store_dist_weight_self_next),state_conv_to_2d_tensor(store_dist_weight_obs_next),torch.tensor(reward,device=device)) 
            
            #print(dist_weight_next[store_min_index_mix_all[1:num_agents+1],0,:])
            #else:
            
           #     Agent_Mem_M.push(state_conv_to_3d_tensor(state),state_conv_to_3d_tensor(current_adjacency_matrix_global),state_conv_to_3d_tensor(v_action[1:num_agents,:]), state_conv_to_3d_tensor(store_next_state),state_conv_to_3d_tensor(store_next_adjacency_matrix_global),\
           #                 state_conv_to_2d_tensor(state_o[1:num_agents,:,:]),state_conv_to_2d_tensor(current_sub_adjacency_matrix[1:num_agents,:,:]),state_conv_to_2d_tensor(store_next_state_o[1:num_agents,:,:]),state_conv_to_2d_tensor(store_next_sub_adjacency_matrix[1:num_agents,:,:]) ,torch.tensor([reward],device=device))
            # if reward <= -0.05:
            #     Agent_Mem_L.push(state_conv_to_3d_tensor(state),state_conv_to_3d_tensor(current_adjacency_matrix_global),state_conv_to_3d_tensor(v_action[1:num_agents,:]), state_conv_to_3d_tensor(store_next_state),state_conv_to_3d_tensor(store_next_adjacency_matrix_global),\
            #                 state_conv_to_2d_tensor(state_o[1:num_agents,:,:]),state_conv_to_2d_tensor(current_sub_adjacency_matrix[1:num_agents,:,:]),state_conv_to_2d_tensor(store_next_state_o[1:num_agents,:,:]),state_conv_to_2d_tensor(store_next_sub_adjacency_matrix[1:num_agents,:,:]) ,torch.tensor([reward],device=device))
            
            # print(len(Agent_Mem_H))
            # print(len(Agent_Mem_M))
            # print(len(Agent_Mem_L))
            #状态更新
            v_1 = v.copy()
            x_1 = x.copy()

            v_obs_1 = v_obs.copy()
            x_obs_1 = x_obs.copy()

            state = next_state.copy()
            state_o = next_state_o.copy()
            state_o_obs = next_state_o_obs.copy()
            #state_o_local = next_state_o_local.copy()
            dist_weight = dist_weight_next.copy()
            dist_weight_self = dist_weight_self_next.copy()
            dist_weight_obs = dist_weight_obs_next.copy()
            #邻接矩阵更新
            current_adjacency_matrix_global = store_next_adjacency_matrix_global.copy()
            current_sub_adjacency_matrix = store_next_sub_adjacency_matrix.copy()
            current_sub_adjacency_matrix_local = next_sub_adjacency_matrix_local.copy()
                #网络训练
            if len(Agent_Mem_predicate)>=BATCH_SIZE_P and counter%10==0 and update_rate_flag:
                transitions_data = Agent_Mem_predicate.sample(int(BATCH_SIZE_P)) 
                # namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_obs','action'))
                batch_pre = Transition_predicate(*zip(*transitions_data))#打包
                state_o_pre_batch = torch.stack(batch_pre.state_o)
                dist_weight_pre_batch = torch.stack(batch_pre.dist_weight)
                state_o_obs_pre_batch = torch.stack(batch_pre.state_o_obs)
                dist_weight_obs_pre_batch = torch.stack(batch_pre.dist_weight_obs)
                action_pre_batch = torch.stack(batch_pre.action)
                predicate_action_mean,predicate_action_std = GCN_actor_predicate(dist_weight_pre_batch,state_o_pre_batch,dist_weight_obs_pre_batch,state_o_obs_pre_batch)
                dist = torch.distributions.Normal(predicate_action_mean, predicate_action_std)
                noise = torch.distributions.Normal(0, 1)
                z_sample = noise.sample()
                predicate_action = torch.tanh(predicate_action_mean+predicate_action_std*z_sample)

                #predicate_action= dist.sample()
                predicate_action=torch.clamp(predicate_action,-1,1)

                loss_predict = F.mse_loss(predicate_action, action_pre_batch, reduction="sum")
                if  counter%40==0:
                    #print('loss_value',td_error)
                    print('train predicate_loss: ',loss_predict)
                GCN_predicate_optimizer.zero_grad()
                loss_predict.backward()
                for param in GCN_actor_predicate.parameters():
                    nn.utils.clip_grad_norm_(param, max_norm=1, norm_type=2)   # 将所有的梯度限制在-1到1之间
                GCN_predicate_optimizer.step()

                temp_loss= (loss_predict.clone()).to('cpu')
                predict_loss = temp_loss.detach().item()

                #update predicate network parameter
            for x_para in GCN_actor_predicate.state_dict().keys():
                eval('GCN_actor_predicate.' + x_para + '.data.mul_((1-TAU_P))')
                eval('GCN_actor_predicate.' + x_para + '.data.add_(TAU_P*GCN_actor_policy.' + x_para + '.data)')

            #plot_flag=True
            if len(Agent_Mem_H) >= int(BATCH_SIZE) and counter%5==0 and TAU_P >0.00000005: #and len(Agent_Mem_M) >= int(BATCH_SIZE/4.0):# and len(Agent_Mem_L) >= int(BATCH_SIZE/4):   #网络训练
                #Network parameter update
                plot_flag=True
                for x_para in GCN_actor_target.state_dict().keys():
                    eval('GCN_actor_target.' + x_para + '.data.mul_((1-TAU_A))')
                    eval('GCN_actor_target.' + x_para + '.data.add_(TAU_A*GCN_actor_policy.' + x_para + '.data)')

                for x_para in GCN_critical_DDPG_target.state_dict().keys():
                    eval('GCN_critical_DDPG_target.' + x_para + '.data.mul_((1-TAU))')
                    eval('GCN_critical_DDPG_target.' + x_para + '.data.add_(TAU*GCN_critic_DDPG_policy.' + x_para + '.data)')   

                for x_para in GCN_critical_DDPG_target_2.state_dict().keys():
                    eval('GCN_critical_DDPG_target_2.' + x_para + '.data.mul_((1-TAU))')
                    eval('GCN_critical_DDPG_target_2.' + x_para + '.data.add_(TAU*GCN_critic_DDPG_policy_2.' + x_para + '.data)')   

                #sampling
                transitions_H = Agent_Mem_H.sample(int(BATCH_SIZE))
                #transitions_M = Agent_Mem_M.sample(int(BATCH_SIZE/4.0))
                #transitions_L = Agent_Mem_L.sample(int(BATCH_SIZE/4))
                        
                batch=Transition(*zip(*transitions_H))#打包
                
                action_batch = torch.cat(batch.action)
                state_o_batch = torch.stack(batch.state_o)
                dist_weight_batch = torch.stack(batch.dist_weight)
                dist_weight_self_batch = torch.stack(batch.dist_weight_self)
                state_o_obs_batch = torch.stack(batch.state_o_obs)
                dist_weight_obs_batch = torch.stack(batch.dist_weight_obs)

                next_state_o_batch = torch.stack(batch.next_state_o)
                next_dist_weight_batch = torch.stack(batch.next_dist_weight)
                next_dist_weight_self_batch = torch.stack(batch.next_dist_weight_self)
                next_state_o_obs_batch = torch.stack(batch.next_state_o_obs)
                next_dist_weight_obs_batch = torch.stack(batch.next_dist_weight_obs)
                #print(next_state_o_batch[0,:,:,:])
                #print(next_sub_adjacency_matrix_bath[0,:,:,:])
                reward_batch = torch.stack(batch.reward)
                

                if  counter%1==0:
                    #actor_action_batch = torch.zeros(BATCH_SIZE,num_agents,2)
                    for agent_i in range(0,num_agents):
                        dat_agent_i=state_o_batch[:,agent_i,:,:]
                        action_policy_mean_batch, action_policy_std_batch =  GCN_actor_policy(dist_weight_batch[:,agent_i],state_o_batch[:,agent_i],dist_weight_obs_batch[:,agent_i:agent_i+1],state_o_obs_batch[:,agent_i])
                        #Behavioral sampling
                        dist = torch.distributions.Normal(action_policy_mean_batch, action_policy_std_batch)
                        noise = torch.distributions.Normal(0, 1)
                        z_sample = noise.sample()
                        action_policy_batch = torch.tanh(action_policy_mean_batch+action_policy_std_batch*z_sample)
                        action_policy_batch=torch.clamp(action_policy_batch,-1,1)
                        action_logprob = dist.log_prob(action_policy_mean_batch+action_policy_std_batch*z_sample)-torch.log(1-action_policy_batch.pow(2)+1e-6)
                        action_logprob_mean = torch.mean(action_logprob,dim=1).unsqueeze(1) #joint distribution
                        
                        actor_action_batch = action_batch.clone()
                        actor_action_batch[:,agent_i,:] = action_policy_batch                         
                        state_action_values_3 = GCN_critic_DDPG_policy(dist_weight_batch,dist_weight_self_batch,state_o_batch,dist_weight_obs_batch,state_o_obs_batch,actor_action_batch,agent_i)
                        state_action_values_4 = GCN_critic_DDPG_policy_2(dist_weight_batch,dist_weight_self_batch,state_o_batch,dist_weight_obs_batch,state_o_obs_batch,actor_action_batch,agent_i)
                        #actor_loss =  0.8*torch.exp(-torch.mean(state_action_values_1.max(state_action_values_2))*1.0)

                        actor_loss =  torch.mean( alpha * action_logprob_mean -(state_action_values_3.min(state_action_values_4)))  
                        #print(state_action_values_1.min(state_action_values_2))
                        if  counter%40==0:
                            #print('loss_value',td_error)
                            print('actor_loss: ',actor_loss)
                        GCN_actor_optimizer.zero_grad()
                        actor_loss.backward()
                        for param in GCN_actor_policy.parameters():
                            nn.utils.clip_grad_norm_(param, max_norm=1, norm_type=2)   # 将所有的梯度限制在-1到1之间
                        GCN_actor_optimizer.step()

                        # alpha update
                        alpha_loss = -torch.mean(log_alpha.exp() * (action_logprob_mean + target_entropy).detach())
                        optimizer_alpha.zero_grad()
                        alpha_loss.backward()
                        optimizer_alpha.step()
                        alpha=log_alpha.exp()

                for agent_k in range(0,num_agents):

                    next_state_action_cat_ddpg = torch.zeros(BATCH_SIZE,num_agents,2).to(device=device)
                    next_action_logprob_mean = torch.zeros(BATCH_SIZE,num_agents,1).to(device=device)
                    for agent_i in range(0,num_agents):
                        next_state_action_mean_batch,next_state_action_std_batch = GCN_actor_target(next_dist_weight_batch[:,agent_i],next_state_o_batch[:,agent_i],next_dist_weight_obs_batch[:,agent_i:agent_i+1],next_state_o_obs_batch[:,agent_i])#+0.05*torch.randn(BATCH_SIZE,2).to(device=device)
                        #print(next_state_action_batch)

                        dist = torch.distributions.Normal(next_state_action_mean_batch,next_state_action_std_batch)
                        noise = torch.distributions.Normal(0, 1)
                        z_sample_next = noise.sample()
                        next_state_action_batch = torch.tanh(next_state_action_mean_batch+next_state_action_std_batch*z_sample_next)
                        next_state_action_batch=torch.clamp(next_state_action_batch,-1,1)
                        next_action_logprob = dist.log_prob(next_state_action_mean_batch+next_state_action_std_batch*z_sample_next)-torch.log(1-next_state_action_batch.pow(2)+1e-6)
                        next_action_logprob_mean[:,agent_i,:] = torch.mean(next_action_logprob,dim=1).unsqueeze(1) #joint distribution

                        next_state_action_cat_ddpg[:,agent_i,:] = next_state_action_batch 
                    
                    #print(next_state_action_cat)

                    next_state_action_values_d = GCN_critical_DDPG_target(next_dist_weight_batch,next_dist_weight_self_batch,next_state_o_batch,next_dist_weight_obs_batch,next_state_o_obs_batch,next_state_action_cat_ddpg, agent_k)
                    #temp_1 = next_dist_weight_self_batch[:,agent_k:agent_k+1,0,:]
                    #reward_culm =torch.matmul( dist_weight_self_batch[:,agent_k:agent_k+1,0,:], reward_batch[:,1:num_agents+1,:] ).squeeze(2)
                    expected_state_action_values_d = (next_state_action_values_d - alpha.detach()*torch.mean(next_action_logprob_mean,dim=1)) * GAMMA_1 + reward_batch[:,0,:] # Using torch.mean(next_action_logprob_mean,dim=1) or using torch.sum(next_action_logprob_mean,dim=1)
                    next_Q_d =  GCN_critic_DDPG_policy(dist_weight_batch,dist_weight_self_batch,state_o_batch,dist_weight_obs_batch,state_o_obs_batch,action_batch, agent_k)

                    td_error_d = F.smooth_l1_loss(expected_state_action_values_d, next_Q_d)    
                    GCN_critical_DDPG_optimizer.zero_grad()
                    td_error_d.backward()
                    if  counter%80==0:
                        print('loss_value DDPG',td_error_d)
                
                    for param in GCN_critic_DDPG_policy.parameters():
                        nn.utils.clip_grad_norm_(param, max_norm=1, norm_type=2)   # 将所有的梯度限制在-1到1之间
                    GCN_critical_DDPG_optimizer.step()

                #critic——2
                transitions_H = Agent_Mem_H.sample(int(BATCH_SIZE))
                #transitions_M = Agent_Mem_M.sample(int(BATCH_SIZE/4.0))
                #transitions_L = Agent_Mem_L.sample(int(BATCH_SIZE/4))
                        
                batch=Transition(*zip(*transitions_H))#打包
                
                action_batch = torch.cat(batch.action)
                state_o_batch = torch.stack(batch.state_o)
                dist_weight_batch = torch.stack(batch.dist_weight)
                dist_weight_self_batch = torch.stack(batch.dist_weight_self)
                state_o_obs_batch = torch.stack(batch.state_o_obs)
                dist_weight_obs_batch = torch.stack(batch.dist_weight_obs)

                next_state_o_batch = torch.stack(batch.next_state_o)
                next_dist_weight_batch = torch.stack(batch.next_dist_weight)
                next_dist_weight_self_batch = torch.stack(batch.next_dist_weight_self)
                next_state_o_obs_batch = torch.stack(batch.next_state_o_obs)
                next_dist_weight_obs_batch = torch.stack(batch.next_dist_weight_obs)
                #print(next_state_o_batch[0,:,:,:])
                #print(next_sub_adjacency_matrix_bath[0,:,:,:])
                reward_batch = torch.stack(batch.reward)
                

                if  counter%1==0:
                    #actor_action_batch = torch.zeros(BATCH_SIZE,num_agents,2)
                    for agent_i in range(0,num_agents):
                        dat_agent_i=state_o_batch[:,agent_i,:,:]
                        action_policy_mean_batch, action_policy_std_batch =  GCN_actor_policy(dist_weight_batch[:,agent_i],state_o_batch[:,agent_i],dist_weight_obs_batch[:,agent_i:agent_i+1],state_o_obs_batch[:,agent_i])
                        #行为抽样
                        dist = torch.distributions.Normal(action_policy_mean_batch, action_policy_std_batch)
                        noise = torch.distributions.Normal(0, 1)
                        z_sample = noise.sample()
                        action_policy_batch = torch.tanh(action_policy_mean_batch+action_policy_std_batch*z_sample)
                        action_policy_batch=torch.clamp(action_policy_batch,-1,1)
                        action_logprob = dist.log_prob(action_policy_mean_batch+action_policy_std_batch*z_sample)-torch.log(1-action_policy_batch.pow(2)+1e-6)
                        action_logprob_mean = torch.mean(action_logprob,dim=1).unsqueeze(1) #joint distribution
                        
                        actor_action_batch = action_batch.clone()
                        actor_action_batch[:,agent_i,:] = action_policy_batch                         
                        state_action_values_3 = GCN_critic_DDPG_policy(dist_weight_batch,dist_weight_self_batch,state_o_batch,dist_weight_obs_batch,state_o_obs_batch,actor_action_batch,agent_i)
                        state_action_values_4 = GCN_critic_DDPG_policy_2(dist_weight_batch,dist_weight_self_batch,state_o_batch,dist_weight_obs_batch,state_o_obs_batch,actor_action_batch,agent_i)
                        #actor_loss =  0.8*torch.exp(-torch.mean(state_action_values_1.max(state_action_values_2))*1.0)

                        actor_loss =  torch.mean( alpha * action_logprob_mean -(state_action_values_3.min(state_action_values_4)))  
                        #print(state_action_values_1.min(state_action_values_2))
                        if  counter%40==0:
                            #print('loss_value',td_error)
                            print('actor_loss: ',actor_loss)
                            print('Entropy',-alpha * action_logprob_mean.mean())
                        GCN_actor_optimizer.zero_grad()
                        actor_loss.backward()
                        for param in GCN_actor_policy.parameters():
                            nn.utils.clip_grad_norm_(param, max_norm=1, norm_type=2)   # 将所有的梯度限制在-1到1之间
                        GCN_actor_optimizer.step()

                        # alpha update
                        alpha_loss = -torch.mean(log_alpha.exp() * (action_logprob_mean + target_entropy).detach())
                        optimizer_alpha.zero_grad()
                        alpha_loss.backward()
                        optimizer_alpha.step()
                        alpha = log_alpha.exp()
                        if  counter%40==0:
                            #print('loss_value',td_error)
                            print('alpha_value: ',alpha)
                        entropy_value = ((alpha.detach() * action_logprob_mean).mean()).detach().item()
                        logpro_value = (( action_logprob_mean).mean()).detach().item()

                for agent_k in range(0,num_agents):

                    next_state_action_cat_ddpg_2 = torch.zeros(BATCH_SIZE,num_agents,2).to(device=device)
                    next_action_logprob_mean_2 = torch.zeros(BATCH_SIZE,num_agents,1).to(device=device)
                    for agent_i in range(0,num_agents):
                        next_state_action_mean_batch,next_state_action_std_batch = GCN_actor_target(next_dist_weight_batch[:,agent_i],next_state_o_batch[:,agent_i],next_dist_weight_obs_batch[:,agent_i:agent_i+1],next_state_o_obs_batch[:,agent_i])#+0.05*torch.randn(BATCH_SIZE,2).to(device=device)
                        #print(next_state_action_batch)

                        dist = torch.distributions.Normal(next_state_action_mean_batch,next_state_action_std_batch)
                        noise = torch.distributions.Normal(0, 1)
                        z_sample_next = noise.sample()
                        next_state_action_batch = torch.tanh(next_state_action_mean_batch+next_state_action_std_batch*z_sample_next)
                        next_state_action_batch=torch.clamp(next_state_action_batch,-1,1)
                        next_action_logprob = dist.log_prob(next_state_action_mean_batch+next_state_action_std_batch*z_sample_next)-torch.log(1-next_state_action_batch.pow(2)+1e-6)
                        next_action_logprob_mean_2[:,agent_i,:] = torch.mean(next_action_logprob,dim=1).unsqueeze(1) #joint distribution

                        next_state_action_cat_ddpg_2[:,agent_i,:] = next_state_action_batch 
                    
                    #print(next_state_action_cat)

                    next_state_action_values_d_2 = GCN_critical_DDPG_target_2(next_dist_weight_batch,next_dist_weight_self_batch,next_state_o_batch,next_dist_weight_obs_batch,next_state_o_obs_batch,next_state_action_cat_ddpg_2, agent_k)
                    #temp_1 = next_dist_weight_self_batch[:,agent_k:agent_k+1,0,:]
                    #reward_culm =torch.matmul( dist_weight_self_batch[:,agent_k:agent_k+1,0,:], reward_batch[:,1:num_agents+1,:] ).squeeze(2)
                    expected_state_action_values_d_2 = (next_state_action_values_d_2 - alpha.detach()*torch.mean(next_action_logprob_mean_2,dim=1)) * GAMMA_1 + reward_batch[:,0,:] # Using torch.mean(next_action_logprob_mean,dim=1) or using torch.sum(next_action_logprob_mean,dim=1)
                    next_Q_d_2 =  GCN_critic_DDPG_policy_2(dist_weight_batch,dist_weight_self_batch,state_o_batch,dist_weight_obs_batch,state_o_obs_batch,action_batch, agent_k)

                    td_error_d_2 = F.smooth_l1_loss(expected_state_action_values_d_2, next_Q_d_2)    
                    GCN_critical_DDPG_optimizer_2.zero_grad()
                    td_error_d_2.backward()
                    if  counter%80==0:
                        print('loss_value DDPG',td_error_d_2)
                        print('Entropy critic',alpha.detach()*torch.mean(next_action_logprob_mean_2))
                    for param in GCN_critic_DDPG_policy_2.parameters():
                        nn.utils.clip_grad_norm_(param, max_norm=1, norm_type=2)   # 将所有的梯度限制在-1到1之间
                    GCN_critical_DDPG_optimizer_2.step()



            run_time=counter*t_gap
            
            # fig = plt.figure(2)

            # title_str='flocking at time t='+str(run_time)+'s'
    
            # #plt.clf()
            # plt.title(title_str)
            # plt.axis([-50, 100, -50, 100])
            # plt.scatter(x_1[:,0],x_1[:,1], color='r', marker='.',s=5)
            # #plt.scatter(x_1[0,0],x_1[0,1], color='b', marker='+')
            # plt.scatter(x_r[0,0],x_r[0,1], color='b', marker='.',s=5)
            # if(num_obs>0):
            #    plt.scatter(x_obs[:,0],x_obs[:,1],color='k', marker='.',s=20) 
            
            # plt.show() 
            # plt.pause(0.1) 
            


            if run_time >= 80 :
                if var>0.08:
                    var=var*0.99
                if GAMMA_1 < 0.8:
                    GAMMA_1 = GAMMA_1+0.005
                if GAMMA_2 < 0.8:
                    GAMMA_2 = GAMMA_2+0.005
                print('var',var)
                print('gmma_1:',GAMMA_1)
                print('gmma_2:',GAMMA_2)
                writer.add_scalar('gamma',GAMMA_1,i_episode)
                writer.add_scalar('var',var,i_episode)
                print('run_time of episode', run_time, current_loss)
                writer.add_scalar('end loss',current_loss,i_episode)
                writer.add_scalar('Entropy',entropy_value,i_episode)
                writer.add_scalar('logpro_value',logpro_value,i_episode)
                if len(test_loss_all)> 0 :
                    writer.add_scalar('predict loss test',sum(test_loss_all)/len(test_loss_all),i_episode)
                    print('Average test loss of each episode:',sum(test_loss_all)/len(test_loss_all))
                average_reward = cul_reward

                if len(average_reward_store) < eval_capacity:
                    average_reward_store.append(None)
                average_reward_store[point] = average_reward
                point = (point + 1) % eval_capacity

                if len(average_reward_store) >= eval_capacity and np.mean(np.array(average_reward_store))>-10 and np.std(np.array(average_reward_store))<5:
                    update_rate_flag = True
                
                if update_rate_flag:
                    writer.add_scalar('predict loss train',predict_loss,i_episode)
                    TAU_P = TAU_P*0.995
                    writer.add_scalar('TAU_P',TAU_P,i_episode)

                eval_reward = eval_reward + average_reward
                print('-------average_reward----:',average_reward)
                cul_reward = 0
                if plot_flag == True and epsilon != 0:
                    #epsilon=0
            #         fig = plt.figure(1)

            #         title_str='average loss at each episode'
    
            #         #plt.clf()
            #         plt.title(title_str)
            #         plt.axis([0, 50000, -60, 0])
            #         plt.scatter(i_episode,average_reward, color='r', marker='+')

            # #plt.scatter(x_r[:,0],x_r[:,1], color='b', marker='o')
            #         plt.show() 
            #         plt.pause(0.1) 
                    writer.add_scalar('reward',average_reward,i_episode)
                
                #print('run_time: ',run_time)
                break
            else:
                counter=counter+1
        
           


        # if i_episode % TARGET_UPDATE == 0:
        #     #GCN_target.load_state_dict(GCN_policy.state_dict())
        #     for x_para in GCN_actor_target.state_dict().keys():
        #         eval('GCN_actor_target.' + x_para + '.data.mul_((1-TAU))')
        #         eval('GCN_actor_target.' + x_para + '.data.add_(TAU*GCN_actor_policy.' + x_para + '.data)')
        #     for x_para in GCN_critical_target.state_dict().keys():
        #         eval('GCN_critical_target.' + x_para + '.data.mul_((1-TAU))')
        #         eval('GCN_critical_target.' + x_para + '.data.add_(TAU*GCN_critical_policy.' + x_para + '.data)')
        if epsilon == 0:
            print('reward with DRL----------',average_reward,'-------------')
            # fig = plt.figure(3)

            # title_str='average loss with DRL'

            # #plt.clf()
            # plt.title(title_str)
            # plt.axis([0, 50000, -80, 0])
            # plt.scatter(i_episode,average_reward, color='r', marker='+')
            # plt.show() 
            # plt.pause(0.1) 

        if i_episode % 10 == 0 :
            epsilon=0
        else:
            epsilon=0.00005
         
        # if i_episode % 50 == 0 and i_episode!=0:
        #     if eval_reward*1.02 > last_eval_reward:
        #         torch.save(GCN_actor_policy,  'best_actor.pth')
        #         torch.save(GCN_critical_policy1,  'best_critical.pth')
        #         last_eval_reward = eval_reward
        #         eval_reward = 0
        #     else:
        #         GCN_actor_policy= torch.load('best_actor.pth')
        #         GCN_critical_policy1 = torch.load('best_critical.pth')
        #         GCN_critical_policy2 = torch.load('best_critical.pth')
        #         eval_reward=0


        if i_episode % 500 == 0:
            mode_name_actor='GCN_actor_target_v4_3_3_UAV_pre_para2_5_0_entropy_15_model3_2_15_2'+str(i_episode)+'.pth'

            mode_name_predicate='GCN_predicate_target1_v4_3_3_UAV_pre_para2_5_0_entropy_15_model3_2_15_2'+str(i_episode)+'.pth'

            torch.save(GCN_actor_policy,  mode_name_actor)    
            torch.save(GCN_actor_predicate,  mode_name_predicate)  

# np.savetxt('MACRL_DGAT_sys_loss_global_2.csv',loss_sys_store,delimiter=',')    
# np.savetxt('MACRL_DGAT_vel_loss_global_2.csv',loss_vel_store,delimiter=',')
# loss_vel_mean=np.mean(loss_vel_store) 
# loss_sys_mean=np.mean(loss_sys_store) 
# print(loss_vel_mean)
# print(loss_sys_mean)
# fig = plt.figure(3)
# title_str='Average system loss'
        
#                 #plt.clf()
# plt.title(title_str)
# plt.axis([0, 100, -1, 1])
                
# plt.plot(loss_sys_mean)
# plt.plot(loss_vel_mean)
                
# plt.show()
# plt.pause(20.1)       
