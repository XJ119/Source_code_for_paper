import csv
from xml.dom import INDEX_SIZE_ERR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import RNN
import random as rd 
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('train_loss_uav/attention_v2Hz_len_3_3') #loss_11 10, loss_10 005, loss_120 020
array_tran=T.ToTensor()
def state_conv_to_3d_tensor(s):
    temp_state=array_tran(s).to(device)
    return temp_state

def state_conv_to_2d_tensor(A):
    temp_A=torch.from_numpy(A).to(device)
    return temp_A

def train_data_At_construct(data, time_len):
    raw_data_len = data.shape[0]
    last_vel_time_data = np.zeros((raw_data_len - time_len +1,time_len,3))
    des_vel_data = data[time_len-1:raw_data_len,3:6]
    acture_vel_data = data[time_len-1:raw_data_len,6:9]
    for k in range(raw_data_len-time_len+1):
        last_vel_time_data[k] = data[k:k+time_len,0:3]
    return last_vel_time_data, des_vel_data , acture_vel_data 

time_len = 3
motion_model = RNN.RNN_net_AT_2(3,3,3,time_len).to(device)
model_net_optimizer = optim.Adam(motion_model.parameters(),lr=0.0001)
batch_size = 512
episode_num = 1001

if __name__ == '__main__':
    file_name0= 'data_01_4_10.csv'
    file_name1= 'data_02_4_10.csv'
    
    train_data0 = (np.loadtxt(open(file_name0),delimiter=",",skiprows=0)/1.5)[10:,:]
    train_data1 = (np.loadtxt(open(file_name1),delimiter=",",skiprows=0)/1.5)[10:,:]
    
    last_vel_0, des_vel_0, acture_vel_0 = train_data_At_construct(train_data0,time_len)
    last_vel_1, des_vel_1, acture_vel_1 = train_data_At_construct(train_data1,time_len)

    last_vel_all = np.vstack((last_vel_0,last_vel_1))

    des_vel_all = np.vstack((des_vel_0,des_vel_1))
    
    acture_vel_all = np.vstack((acture_vel_0,acture_vel_1))
    

    last_vel = last_vel_all[0:35000,:]
    des_vel = des_vel_all[0:35000,:]
    acture_vel = acture_vel_all[0:35000,:]

    val_last_vel = last_vel_all[35000:,:]
    val_des_vel = des_vel_all[35000:,:]
    val_acture_vel = acture_vel_all[35000:,:]

    len_data = acture_vel.shape[0]
    best_test_loss = -1000
    for episode_i in range(episode_num):
        print('----------epospde-------',episode_i)
        idx = rd.sample(range(len_data),len_data)
        last_vel_mix = state_conv_to_2d_tensor(last_vel[idx])
        des_vel_mix = state_conv_to_2d_tensor(des_vel[idx])
        acture_vel_mix = state_conv_to_2d_tensor(acture_vel[idx])

        epoch_num = int(len_data/batch_size)
        all_bath = range(epoch_num)
        train_bathes = all_bath[0:epoch_num-2]
        test_bathes = all_bath[epoch_num-2:epoch_num]

        all_loss = 0
        test_all_loss = 0
        train_count = 0
        test_count = 0
        for epoch_i in train_bathes:
            data_state = last_vel_mix[epoch_i*batch_size:(epoch_i+1)*batch_size]
            data_input = des_vel_mix[epoch_i*batch_size:(epoch_i+1)*batch_size]
            data_label = acture_vel_mix[epoch_i*batch_size:(epoch_i+1)*batch_size]

            motion_model_data = motion_model(data_state,data_input)
            train_loss = F.smooth_l1_loss(motion_model_data, data_label)  
            model_net_optimizer.zero_grad()
            train_loss.backward()
            model_net_optimizer.step()
            all_loss = all_loss + train_loss.item()*batch_size
            train_count += batch_size
        writer.add_scalar('average train loss',all_loss*1.0/train_count ,episode_i)
        print('train avarage loss:',all_loss*1.0/train_count)
        with torch.no_grad():
            for epoch_j in test_bathes:
                test_state = last_vel_mix[epoch_j*batch_size:(epoch_j+1)*batch_size]
                test_input = des_vel_mix[epoch_j*batch_size:(epoch_j+1)*batch_size]
                test_label = acture_vel_mix[epoch_j*batch_size:(epoch_j+1)*batch_size]
                motion_model_test = motion_model(test_state,test_input)
                test_loss = F.smooth_l1_loss(motion_model_test, test_label) 
                test_all_loss = test_all_loss + test_loss.item()*batch_size
                test_count = test_count + batch_size
            writer.add_scalar('average test loss',test_all_loss*1.0/test_count ,episode_i)
            print('test avarage loss:',test_all_loss*1.0/test_count)
            if math.fabs(test_all_loss*1.0/test_count) < math.fabs(best_test_loss):
                best_test_loss = test_all_loss*1.0/test_count
                mode_name='train_motion_model_best_uav_attention_2Hz_len3_3_2.pth'
                torch.save(motion_model,mode_name)
            if episode_i%100 ==0:
                mode_name='train_motion_model_uav_412_attention_2Hz_len3_3_2'+str(episode_i)+'.pth'
                torch.save(motion_model,mode_name)

        with torch.no_grad():
            validation_state = state_conv_to_2d_tensor(val_last_vel)
            validation_input = state_conv_to_2d_tensor(val_des_vel)
            validation_label = state_conv_to_2d_tensor(val_acture_vel)
            motion_model_val = motion_model(validation_state,validation_input)
            val_loss = F.smooth_l1_loss(motion_model_val, validation_label) 

            writer.add_scalar('average validation loss',val_loss.item(),episode_i)  # timelen = 4 test avarage loss: 0.0001380651819681094 validation avarage loss: 0.00024657183192842736
            print('validation avarage loss:',val_loss.item())                       # timelen = 2 test avarage loss: 9.784250507562965e-05 validation avarage loss: 0.0001708099470794598
                                                                                    # timelen = 1 test avarage loss: 8.643290139300062e-05 validation avarage loss: 0.00014772136653746725
                                                                                    # attention v2 :  timelen = 1 test avarage loss: 8.643290139300062e-05 validation avarage loss: 0.00014772136653746725
                                                                                    # attention v2 :  timelen = 2 test avarage loss: 7.668586147108793e-05 validation avarage loss: 0.00013837768329750082
 
                                                                                    # attention v2 :  timelen = 4 test avarage loss: 7.489285670365833e-05 validation avarage loss: 0.00014096165449623327
                                                                                    # attention v2 :  timelen = 5 test avarage loss: 7.958099723457802e-05 validation avarage loss: 0.0001417230192313894