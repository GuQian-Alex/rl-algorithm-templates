"""
Vanilla REINFORCE for discrete actions on CartPole-v1.

This version is implemented by myself, and remains vanilla REINFORCE, no learned baseline, no reward-to-go modify.

The aim of this version is to help me remember my practice in writing RL code.
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import  Categorical
from torch.optim import Adam


def network(input_dim,output_dim,hidden_size,activation_func=nn.Tanh):
    layers=[]
    input_layer = nn.Linear(input_dim,hidden_size[0],dtype=torch.float32)
    output_layer = nn.Linear(hidden_size[-1],output_dim,dtype=torch.float32)
    layers.append(input_layer)
    layers.append(activation_func())
    for i in range(len(hidden_size)-1):
        layer = nn.Linear(hidden_size[i],hidden_size[i+1],dtype=torch.float32)
        activation = activation_func()
        layers.append(layer)
        layers.append(activation)
    layers.append(output_layer)
    layers.append(nn.Identity())
    return nn.Sequential(*layers)

def forward(network,o):
    logits = network(o)
    m = Categorical(logits=logits)
    return m

@torch.no_grad()
def get_action(network,obs):
    dist = forward(network,obs)
    action = dist.sample()
    return action.item()

def reinforce(env_name="CartPole-v1",lr=0.001,epoch=5000):

    env =gym.make(env_name)
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    policy = network(obs_dim,act_dim,[20])
    optimizer = Adam(policy.parameters(),lr=lr)


    for i in range(epoch):
        obs,_ = env.reset() 
        ep_rew = []
        ep_act = []
        ep_obs = []
    
        while True:

            act = get_action(policy,torch.as_tensor(obs,dtype=torch.float32))
            next_obs,reward,terminated,truncated,info = env.step(act)
            done = terminated or truncated
            ep_rew.append(reward)
            ep_act.append(act)
            ep_obs.append(obs)
            obs = next_obs

            if done :
                rew = sum(ep_rew)
                ep_ret = [rew]*len(ep_rew)
                break


        ep_obs = torch.as_tensor(np.asarray(ep_obs),dtype=torch.float32)
        ep_act = torch.as_tensor(ep_act,dtype=torch.long)
        ep_ret = torch.as_tensor(ep_ret,dtype=torch.float32)
    
        m = forward(policy,ep_obs)
        log_prob = m.log_prob(ep_act)

        optimizer.zero_grad()
        loss = -(log_prob*ep_ret).sum()
        loss.backward()
        optimizer.step()
        print("epoch ",i,"finished,ep_return is:",ep_ret[0].item())
    

    env.close()





if __name__ == "__main__":

    # hs = [20,30]
    # net = network(10,10,hs)
    # print(net)

    reinforce()
    

        
