# Source_code_for_paper
This code is for the paper named "A Model Learning Based Multi-agent Flocking Collaborative Control Method for Stochastic Communication Environment"
This version of the code is a preliminary  version, with a lot of redundant code and incomplete annotations in the code. In the next version, we will make updates.
We declare that the provided code is for research purposes only and cannot involve commercial activities.

The file "motion_UAV_model_attention_predict_2_test_722_train_SAC_modellen3_train.py" is the policy training python file.
The file "motion_UAV_model_attention_predict_2_test_722_train_SAC_modellen3_train.py" is the policy test file.
  In the training, the number of the agent is set to 6, the number of obstacle is set to 3,  the number of training episodes is 4000.
  In environmental interaction and policy training, the actions of agents come from the sampling of policy network.
  In the test,  the mean of actor network is set to the action of agent.

The file "GCN_actor_target_v4_3_3_UAV_pre_para2_5_0_entropy_15_model3XX.pth" is the policy model trained in SANN-3 motion model. XX represents the corresponding model in the XX episode.
The file "GCN_predicate_target1_v4_3_3_UAV_pre_para2_5_0_entropy_15_model3XX.pth" is the behavior reasoning model.
The file "train_motion_model_uav_412_attention_2Hz_len3_21000.pth" is a SANNN-3 motion model.

The "flocking_run" folder stores the training data by using Tensorboard.
"RNN.py" describes the SANNN network.
"GCN_4_obs_DDPG_v3_dist_attation_reward.py" in the "class_pack" folder describes the policy network, critic network and behavior reasoning network.


