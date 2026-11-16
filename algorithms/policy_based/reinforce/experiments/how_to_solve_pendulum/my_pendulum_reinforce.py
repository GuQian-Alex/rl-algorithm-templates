'''
Original version of vanilla REINFORCE playing Pendulum-v0
Reward-to-go and batch-trajectory training
One network update cost nearly 17s,with about 17s for sampling and 0.13s for training

'''

import gymnasium as gym
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import Parameter
from torch.optim import Adam
import torch
import numpy as np


def network(sizes,activation = nn.Tanh,output_activation = nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i<len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i],sizes[i+1]),act()]
    return nn.Sequential(*layers)


def get_distribution(net,obs,std):
    mu = net(obs)
    m = Normal(mu,std)
    return m

def get_action(net,obs,std):
    dist = get_distribution(net,obs,std)
    action = dist.sample()
    return action

def reward_to_go(rew):
    rtg = np.zeros_like(rew)
    for i in reversed(range(len(rew))):
        rtg[i] = rew[i]+rtg[i+1] if i < len(rew)-1 else rew[i]
    return rtg

def train(env_name = "Pendulum-v1",lr = 0.001,batch_size = 500,epochs = 10000,hidden_sizes=[64,64]):
    env = gym.make(env_name)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    policy = network([obs_dim]+hidden_sizes+[act_dim])
    log_std = Parameter(torch.zeros(act_dim))
    optimizer = Adam(list(policy.parameters())+[log_std],lr)

    for i in range(epochs):
        batch_obs = []
        batch_act = []
        batch_ret = []
        obs,_ = env.reset()
        ep_rew = []
        batch_ep_ret = []

        while True:
            obs = torch.as_tensor(np.asarray(obs),dtype=torch.float32)
            act = get_action(policy,obs,log_std.exp()).cpu().numpy()
            next_obs,reward,terminated,truncated,_ = env.step(act)
            done = terminated or truncated
            batch_obs.append(obs)
            batch_act.append(act)
            ep_rew.append(reward)
            obs = next_obs

            if done:
                # ret = sum(ep_rew)
                # batch_ret += [ret]*len(ep_rew)
                rtg = reward_to_go(ep_rew)
                batch_ret.extend(rtg)
                batch_ep_ret.append(sum(ep_rew))

                ep_rew = []
                obs,_ = env.reset()


                if len(batch_obs)/200 >= batch_size:
                    break

        batch_obs = torch.as_tensor(np.asarray(batch_obs),dtype=torch.float32)
        batch_act = torch.as_tensor(np.asarray(batch_act),dtype=torch.float32) 
        batch_ret = torch.as_tensor(batch_ret,dtype=torch.float32)

        logprob = get_distribution(policy,batch_obs,log_std.exp()).log_prob(batch_act).sum(-1)

        optimizer.zero_grad()
        loss = -(logprob*batch_ret).sum()/batch_size
        loss.backward()
        optimizer.step()
        print("epoch",i,"finished, return is :",np.mean(batch_ep_ret))

    env.close()


if __name__ == "__main__":
    train()





