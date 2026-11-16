'''
This is continuous version of REINFORCE for playing InvertedPendulum-v5
Use reward-to-go for the actor; keep the learned TD baseline unchanged.
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


def reward_to_go(rewards):
    """每一步到本条轨迹结束的累计奖励，gamma=1，不修改输入。"""
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1:
        raise ValueError("Rewards must have shape [T]")
    returns = np.empty_like(rewards)
    running_return = np.float32(0.0)
    for t in range(len(rewards) - 1, -1, -1):
        running_return = rewards[t] + running_return
        returns[t] = running_return
    return returns


def train(env_name = "InvertedPendulum-v5",lr=0.001,epochs=5000,hidden_size = [32,32]):
    env= gym.make(env_name)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    policy = network([obs_dim]+hidden_size+[act_dim])
    critic = network([obs_dim]+hidden_size+[1])

    #易错点，这里一般是用Parameter来输出log_std，真正使用的时候还需要exp取指数
    log_std = Parameter(torch.zeros(act_dim))

    #注意这里的Adam参数输入的方法
    optimizer1 = Adam(list(policy.parameters())+[log_std],lr)
    optimizer2 = Adam(critic.parameters(),lr)

    for i in range(epochs):
        ep_obs = []
        ep_act = []
        ep_rew = []
        ep_next_obs = []
        ep_terminated = []
        obs,_ = env.reset()

        while True:
            #易错点，对于cotinuous action来说，采样获取action之后是tensor，需要用.cpu().numpy()来转换成环节可接收的数组
            act = get_action(policy,torch.as_tensor(obs,dtype=torch.float32),log_std.exp()).cpu().numpy()
            
            next_obs,reward,terminated,truncated,_=env.step(act)
            ep_next_obs.append(next_obs)
            done = terminated or truncated
            ep_terminated.append(terminated)
            ep_obs.append(obs)
            ep_act.append(act)
            ep_rew.append(reward)
            obs = next_obs

            if done:
                ret = sum(ep_rew)
                #Actor 改用 G_t；TimeLimit 截断时只累计已采样的有限轨迹奖励。
                ep_ret = reward_to_go(ep_rew)
                break

        #易错点，在交互采样的得到的数据必须转换成tensor之后才能用网络进行训练
        ep_ret = torch.as_tensor(ep_ret,dtype=torch.float32)
        ep_obs = torch.as_tensor(np.asarray(ep_obs),dtype=torch.float32)
        ep_act = torch.as_tensor(np.asarray(ep_act),dtype=torch.float32)
        ep_next_obs = torch.as_tensor(np.asarray(ep_next_obs),dtype=torch.float32)
        ep_rew = torch.as_tensor(ep_rew,dtype=torch.float32)
        ep_terminated = torch.as_tensor(ep_terminated,dtype=torch.float32)
        value_baseline = critic(ep_obs).squeeze(-1).detach()
        optimizer1.zero_grad()
        #有易错点，对于连续
        logprob = get_distribution(policy,ep_obs,log_std.exp()).log_prob(ep_act).sum(dim=-1)
        
        loss1= - (logprob*(ep_ret-value_baseline)).sum()
        loss1.backward()
        optimizer1.step()

        #TD critic 保持原样：真正终止不 bootstrap，TimeLimit 截断继续 bootstrap。
        optimizer2.zero_grad()
        td_target = (ep_rew + (1-ep_terminated)*critic(ep_next_obs).squeeze(-1)).detach()
        loss2 = ((td_target - critic(ep_obs).squeeze(-1))**2).mean()
        loss2.backward()
        optimizer2.step()

        
        print("epoch",i,"finished, return is :",ret)

    env.close()


if __name__ == "__main__":
    train()
