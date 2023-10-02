import numpy as np
import math
import random
class Anti_Fn():
  #  def __init__(self):
    def action_select_space(self,I,J,action_space):
        if I==0 and J==0:
            var_space=[0,1,7,8]
        elif I==0 and J>0 and J<7:
            var_space=[0,1,5,6,7,8]
        elif I==0 and J==7:
            var_space=[0,5,6,7]
        elif I>0 and I<7 and J==7:
            var_space=[0,3,4,5,6,7]
        elif I==7 and J==7:
            var_space=[0,3,4,5]
        elif J>0 and J<7 and I==7:
            var_space=[0,1,2,3,4,5]
        elif I==7 and J==0:
            var_space=[0,1,2,3]
        elif I>0 and I<7 and J==0:
            var_space=[0,1,2,3,7,8]
        elif I>0 and I<7 and J>0 and J<7:
            var_space=action_space
        return var_space
    def action_select_s(self,var_select_action_temp,state_gamma_info,J_g,I_g):
        action_len=len(var_select_action_temp)
        temp_time=np.ones((9,1))
        for s in range(action_len):
            if var_select_action_temp[s]==0:
                temp_J=J_g
                temp_I=I_g
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==1:
                temp_I=I_g+1
                temp_J=J_g
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==2:
                temp_J=J_g-1
                temp_I=I_g+1
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==3:
                temp_J=J_g-1
                temp_I=I_g
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==4:
                temp_J=J_g-1
                temp_I=I_g-1
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==5:
                temp_J=J_g
                temp_I=I_g-1
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==6:
                temp_J=J_g+1
                temp_I=I_g-1
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==7:
                
                temp_J=J_g+1
                temp_I=I_g
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            elif var_select_action_temp[s]==8:
                temp_J=J_g+1
                temp_I=I_g+1
                temp_time[int(var_select_action_temp[s])]=state_gamma_info[int(temp_J),int(temp_I)]
            else:
                pass
        #np.where(obs_nbr[:,a]==1)
        #value=np.min(temp_time)
        #index=np.argmin(temp_time)
        var_action=np.where(temp_time==0)
        var_action_temp=var_action[0]
        if len(var_action_temp)==0:
            index_sub=np.where(state_gamma_info==0)
            index_sub_line=index_sub[0]
            index_sub_col=index_sub[1]
            if len(index_sub_line)==0:
                return np.array(var_select_action_temp)
            else:
                temp_target=np.zeros((2,1))
                temp_dis=1000
                for kk in range(len(index_sub_line)):
                    y=int(index_sub_line[kk])
                    x=int(index_sub_col[kk]) 
                    dis=np.sqrt((y-J_g)**2+(x-I_g)**2)
                    if dis<temp_dis:
                        temp_dis=dis
                        temp_target[0]=y
                        temp_target[1]=x
                if temp_target[0]-J_g>0 and temp_target[1]-I_g>0:
                    var_action_temp=np.array([8])
                elif temp_target[0]-J_g==0 and temp_target[1]-I_g>0:
                    var_action_temp=np.array([1])
                elif temp_target[0]-J_g<0 and temp_target[1]-I_g>0:
                    var_action_temp=np.array([2])
                elif temp_target[0]-J_g>0 and temp_target[1]-I_g==0:
                    var_action_temp=np.array([7])
                elif temp_target[0]-J_g>0 and temp_target[1]-I_g<0:
                    var_action_temp=np.array([6])
                elif temp_target[0]-J_g==0 and temp_target[1]-I_g<0:
                    var_action_temp=np.array([5])
                elif temp_target[0]-J_g<0 and temp_target[1]-I_g<0:
                    var_action_temp=np.array([4])
                elif temp_target[0]-J_g<0 and temp_target[1]-I_g==0:
                    var_action_temp=np.array([3])
                else:
                    var_action_temp=np.array([0])


                return var_action_temp

        else:
            return var_action_temp

        

    def bump_func_2(self,z,h1,h2):
        ref=100.0
        if z>h1:
            p_h=0.0
        elif z<=h1 and z>h2:
            p_h =ref* 0.5*(1+math.cos(math.pi*(z-h2)/(h1-h2)))
        else:
            p_h=ref
        return p_h
        

    def get_gap_temp(self,r):
        r_length=(r.shape)[0]
        gap=np.zeros((r_length,r_length,2))
        for a in range(2,r_length+1):
            for b in range(1,a):
                gap[a-1,b-1,:]=r[b-1,:]-r[a-1,:]
        print(gap[:,:,0])
        print(gap[:,:,1])

        gap[:,:,0]=gap[:,:,0]-gap[:,:,0].transpose()
        gap[:,:,1]=gap[:,:,1]-gap[:,:,1].transpose()
        for a in range(2,r_length):
            for b in range(2,r_length):
                gap[a-1,b-1,:]=gap[a-1,b-1,:]/2
        print(gap[:,:,0])
        return gap

    def get_gap(self,r):
        r_length=(r.shape)[0]
        gap=np.zeros((r_length,r_length,2))
        for a in range(2,r_length+1):
            for b in range(1,a):
                gap[a-1,b-1,:]=r[b-1,:]-r[a-1,:]
     #   print(gap[:,:,0])
     #   print(gap[:,:,1])

        gap[:,:,0]=gap[:,:,0]-gap[:,:,0].transpose()
        gap[:,:,1]=gap[:,:,1]-gap[:,:,1].transpose()
       # print(gap[:,:,0])
        return gap

    def squd_norm(self,z):
      
        if len(z.shape)==2:
            s_n = z**2
            return s_n
        else:
            s_n = z[:,:,0]**2 + z[:,:,1]**2
            return s_n

    def action_func_1(self,z,d,c1):
        z=z/d   
        
        p_h=-(c1*0.5*math.pi/d)*np.sin(0.5*math.pi*(z+1))
        
        return p_h


    def flock_func_temp(self,z,d,c1):
        z_temp=z/d   
        z_dim=z.ndim
        if z_dim==1:
            p_h=-(c1*0.5*math.pi/d)*np.sin(0.5*math.pi*(z+1))
        else:
            p_h=np.zeros((z.shape))
            for i in range(0,z.shape[0]):
                for j in range(0,z.shape[0]):
                    
                    if z_temp[i,j] >= 1:
                        data= z[i,j] - d
                        p_h[i,j] = (data/1.0)*(data/1.0)*(data/1.0)
                        if p_h[i,j]>10:
                            p_h[i,j]=10
                    else:
                        data= z[i,j] - d
                        p_h[i,j] = 1.0*(data/1.0)*(data/1.0)*(data/1.0)*(data/1.0)  
                    #p_h[i,j]= -(c1*0.5*math.pi/d)*np.sin(0.5*math.pi*(z[i,j]+1))
        return p_h

    def flock_func(self,z,d,c1):
        z_temp=z/d   
        z_dim=z.ndim
        dist_adj = np.zeros((z.shape))
        if z_dim==1:
            p_h=-(c1*0.5*math.pi/d)*np.sin(0.5*math.pi*(z+1))
        else:
            p_h=np.zeros((z.shape))
            
            for i in range(0,z.shape[0]):
                for j in range(0,z.shape[0]):
                    
                    if z_temp[i,j] >= 1:
                        data= z[i,j] - d
                        p_h[i,j] = (data/1.0)*(data/1.0)                     

                    else:
                        data = z[i,j] - d
                        p_h[i,j] = 1.0*(data/1.0)*(data/1.0) 
                    if (i == j):
                        p_h[i,j] =0.1

                    if p_h[i,j] < 2.0:
                        dist_adj[i,j] = 1.0
                    else:
                        dist_adj[i,j] = 2.0
                    #p_h[i,j]= -(c1*0.5*math.pi/d)*np.sin(0.5*math.pi*(z[i,j]+1))
        return p_h,p_h

    def update_individual_record(self,cell_map,map_pos,x,T,r_s):
        size_x=x.shape[0]
        for r in range(0,size_x):
            other_dist = np.sqrt((x[r,0]-map_pos[:,:,0])**2+(x[r,1]-map_pos[:,:,1])**2)
            temp_map = cell_map[:,:,2*r].copy()
            temp_ind = cell_map[:,:,2*r+1].copy()
            temp_map[other_dist<=r_s]=T
            temp_ind[other_dist<=r_s]=r+1
            cell_map[:,:,2*r]=temp_map.copy()
            cell_map[:,:,2*r+1]=temp_ind.copy()
        return cell_map

    def fuse_record(self,cell_map,nbr):
        nbr_size=nbr.shape[0]
        for i in range(2,nbr_size+1):
            for j in range(1,i):
                if nbr[i-1,j-1]==1:
                    max_time = np.maximum(cell_map[:,:,2*i-2],cell_map[:,:,2*j-2])
                    #print('max_time:',max_time)
                    temp_cell_i = cell_map[:,:,2*i-1].copy()
                    temp_cell_i[cell_map[:,:,2*i-2] != max_time] = j
                    cell_map[:,:,2*i-2] = max_time.copy()
                    cell_map[:,:,2*j-2] = max_time.copy()
                    cell_map[:,:,2*i-1] = temp_cell_i.copy()
                    cell_map[:,:,2*j-1] = temp_cell_i.copy()
        return cell_map
    def fuse_gamma_state(self,state_map,nbr):
        nbr_size=nbr.shape[0]
        for i in range(2,nbr_size+1):
            for j in range(1,i):
                if nbr[i-1,j-1]==1:
                    max_state = np.maximum(state_map[:,:,i-1],state_map[:,:,j-1])
                    #print('max_time:',max_time)
                   
                    state_map[:,:,i-1] = max_state.copy()
                    state_map[:,:,j-1] = max_state.copy()
                    
        return state_map

    


    def fuse_all_records(self,cell_map,num_agents,fused_scan_record):
        max_time = np.maximum(fused_scan_record[:,:,0],cell_map[:,:,0])
        temp_fused = fused_scan_record[:,:,1].copy()
        temp_fused[max_time!=fused_scan_record[:,:,0]] = 1
        fused_scan_record[:,:,1] = temp_fused.copy()  
        fused_scan_record[:,:,0] = max_time.copy()    

        for i in range(3,num_agents*2+1,2):
            max_time = np.maximum(fused_scan_record[:,:,0],cell_map[:,:,i-1])
            temp_fused = fused_scan_record[:,:,1].copy()
            temp_fused[max_time!=fused_scan_record[:,:,0]] = (i+1)/2
            fused_scan_record[:,:,1] = temp_fused.copy()
            fused_scan_record[:,:,0] = max_time.copy()

        return fused_scan_record
    
    def limit(self, v ,v2):
        v_size_0=v.shape[0]
        v_size_1=v.shape[1]
        output_v=np.zeros((v_size_0,v_size_1))
        for i in range(0,v_size_0):
            c=np.linalg.norm(v[i,:])/v2
            if c>1:
                output_v[i,:]=v[i,:]/c
            else:
                output_v[i,:]=v[i,:]
        return output_v
    def limit_state(self, v ,v2):
        v_size_0=v.shape[1]
        output_v=np.zeros((1,v_size_0))
        
        c=np.linalg.norm(v)/v2
        if c>1:
            output_v=v/c
        else:
            output_v=v.copy()
        return output_v

    def limit_v(self, v ,v2):
        v_size_0=v.shape[0]
        output_v=np.zeros((1,v_size_0))
        
        c=np.linalg.norm(v)/v2
        if c>1:
            output_v=v/c
        else:
            output_v=v.copy()
        return output_v

    def action_normalize(self,v,v_max):
        temp_v=v.copy()
        length=v.shape[0]
        for num_i in range(length):
            temp_v[num_i]=temp_v[num_i]/v_max #归一化（-1，1）
            #temp_v[num_i]=(temp_v[num_i]+1)/2 #归一化（0，1）
            if temp_v[num_i,0]>=1:
                temp_v[num_i,0]=1
            if temp_v[num_i,1]>=1:
                temp_v[num_i,1]=1  
            if temp_v[num_i,0]<=-1:
                temp_v[num_i,0]=-1 
            if temp_v[num_i,1]<=-1:
                temp_v[num_i,1]=-1
        return temp_v

    def distance_normalize(self,pos,index,i,r_d):
        #i_index=np.where(index==i)
        temp_td=pos-pos[0]
        length=pos.shape[0]
        for num_i in range(length):
            temp_td[num_i]=self.limit_v(temp_td[num_i]/(r_d),1.0) #归一化（-1，1）
            #temp_td[num_i]=(temp_td[num_i]+1)/2 #归一化（0，1）

        return temp_td

    def distance_normalize_global(self,pos,r_d):
        #center=np.zeros((2,1))
        center = pos[0]
        temp_pos = pos.copy()
        temp_td=temp_pos-center
        length=temp_pos.shape[0]
        for num_i in range(length):
            temp_td[num_i]=self.limit_v(temp_td[num_i]/r_d,1.0)#归一化（-1，1）
            #temp_td[num_i]=(temp_td[num_i]+1)/2 #归一化（0，1）
            #阈值限制先保留

        return temp_td

    def co_vel_normalize(self,pos,index,i,r_d):
        i_index=np.where(index==i)
        temp_td=pos-pos[i_index[0]]
        length=pos.shape[0]
        for num_i in range(length):
            temp_td[num_i]=self.limit_v(temp_td[num_i]/(2.0*r_d),1.0) #归一化（-1，1）
            #temp_td[num_i]=(temp_td[num_i]+1)/2 #归一化（0，1）

        return temp_td

    def co_vel_normalize_1(self,pos,index,i,r_d):
        #i_index=np.where(index==i)
        temp_td=pos - pos[0]
        length=pos.shape[0]
        for num_i in range(length):
            temp_td[num_i]=self.limit_v(temp_td[num_i]/(2.0*r_d),1.0) #归一化（-1，1）
            #temp_td[num_i]=(temp_td[num_i]+1)/2 #归一化（0，1）
            

        return temp_td

    def co_vel_normalize_1_global(self,vel,v_max):
        #center=np.zeros((2,1))
        #center = np.mean(vel, axis=0)
        temp_vel=vel.copy()
        temp_td=temp_vel - vel[0]
        length=vel.shape[0]
        for num_i in range(length):
            temp_td[num_i]=self.limit_v(temp_td[num_i]/(2.0*v_max),1.0) #归一化（-1，1）
            #temp_td[num_i]=(temp_td[num_i]+1)/2 #归一化（0，1）
           

        return temp_td

    
    def action_encode(self,v,bits):
        v_x=v[0]
        v_y=v[1]
        if v_x>=0.99:
            v_x=0.99
        if v_y>=0.99:
            v_y=0.99
        list_res_x=np.zeros((1,bits))
        list_res_y=np.zeros((1,bits))
        tem_v_x=int(v_x*(2**bits))
        tem_v_y=int(v_y*(2**bits))
        return tem_v_y*2**bits + tem_v_x

    def action_decode(self, action,bits):
        decode_data=np.array([0.0,0.0])
        high_data= float(int(action/(2**bits)))
        low_data= float(int(action%(2**bits)))
        decode_data[0]= low_data/float(2**bits)
        decode_data[1]= high_data/float(2**bits)
        decode_data[0]=(decode_data[0]*2)-1
        decode_data[1]=(decode_data[1]*2)-1
        return decode_data



    def one_hot(self,v,bits):#将行为转成二进制编码
        v_x=v[0]
        v_y=v[1]
        if v_x>=0.99:
            v_x=0.99
        if v_y>=0.99:
            v_y=0.99
        list_res_x=np.zeros((1,bits))
        list_res_y=np.zeros((1,bits))
        tem_v_x=int(v_x*(2**bits))
        tem_v_y=int(v_y*(2**bits))
        bits_list=[i for i in range(bits)]#
        reverse_list=bits_list[::-1]
        temp_data_x=tem_v_x
        temp_data_y=tem_v_y
        for i in reverse_list:
            list_res_x[0,i]= int(temp_data_x/(2**i))
            list_res_y[0,i]= int(temp_data_y/(2**i))
            if list_res_x[0,i]==1:
                temp_data_x=temp_data_x-2**i
            if list_res_y[0,i]==1:
                temp_data_y=temp_data_y-2**i
        #print(list_res_y)
        #print(list_res_x)
        return np.hstack((list_res_x,list_res_y))





        


if __name__ == '__main__':

    cnn = Anti_Fn()
    v_data=np.array([[0.5,0.675],[0,1]])
    #v_data_1=[0,1,2,3]
   # v_norm=cnn.action_normalize(v_data,1.0)
    #print('v_norm:',v_norm)
    #v_code=cnn.action_encode(v_norm,4)
    #print('v_code:',v_code)
    v_decode=cnn.action_decode(118,4)
    print('v_decode:',v_decode)

    #print(encode_data)
    #print(v_data_1[::-1])
