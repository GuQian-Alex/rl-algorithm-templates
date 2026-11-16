'''
This is continuous version of REINFORCE for playing InvertedPendulum-v5
Pure vanilla REINFORCE 
'''

import torch
from torch.distributions import Normal
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym
from torch.nn import Parameter
import numpy as np



def network(sizes,activation = nn.Tanh,output_activation = nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i],sizes[i+1]),act()]
    return nn.Sequential(*layers)


def get_distribution(policy,obs,std):
    mu = policy(obs)
    return Normal(mu,std)

@torch.no_grad()
def get_action(policy, obs, std):
    dist = get_distribution(policy, obs, std)
    action = dist.sample()
    return action


def train(env_name = "InvertedPendulum-v5",lr=0.001,epochs=5000,hidden_size = [32,32]):
    env= gym.make(env_name)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    policy = network([obs_dim]+hidden_size+[act_dim])

    #易错点，这里一般是用Parameter来输出log_std，真正使用的时候还需要exp平方处理
    log_std = Parameter(torch.zeros(act_dim))

    #注意这里的Adam参数输入的方法
    optimizer = Adam(list(policy.parameters())+[log_std],lr)

    for i in range(epochs):
        ep_obs = []
        ep_act = []
        ep_rew = []
        obs,_ = env.reset()

        while True:
            #易错点，对于cotinuous action来说，采样获取action之后是tensor，需要用.cpu().numpy()来转换成环节可接收的数组
            act = get_action(policy,torch.as_tensor(obs,dtype=torch.float32),log_std.exp()).cpu().numpy()
            
            next_obs,reward,terminated,truncated,_=env.step(act)
            done = terminated or truncated
            ep_obs.append(obs)
            ep_act.append(act)
            ep_rew.append(reward)
            obs = next_obs

            if done:
                ret = sum(ep_rew)
                ep_ret = [ret]*len(ep_obs)
                break

        #易错点，在交互采样的得到的数据必须转换成tensor之后才能用网络进行训练
        ep_ret = torch.as_tensor(ep_ret,dtype=torch.float32)
        ep_obs = torch.as_tensor(np.asarray(ep_obs),dtype=torch.float32)
        ep_act = torch.as_tensor(np.asarray(ep_act),dtype=torch.float32)
        optimizer.zero_grad()

        #有易错点，对于连续
        logprob = get_distribution(policy,ep_obs,log_std.exp()).log_prob(ep_act).sum(dim=-1)

        loss= - (logprob*ep_ret).sum()
        loss.backward()
        optimizer.step()
        print("epoch",i,"finished, return is :",ret)

    env.close()


if __name__ == "__main__":
    train()
