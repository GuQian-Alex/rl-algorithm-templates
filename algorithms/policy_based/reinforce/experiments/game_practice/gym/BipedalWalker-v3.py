import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.distributions import Normal
from torch.nn import Parameter
import gymnasium as gym


def network(sizes,activation=nn.Tanh):
    layers = []
    for i in range(len(sizes)-1):
        act_func = activation if i < len(sizes)-2 else nn.Identity
        layers += [nn.Linear(sizes[i],sizes[i+1]),act_func()]
    return nn.Sequential(*layers)


def get_distribution(policy,obs,std):
    mu = policy(obs)
    m = Normal(mu,std)
    return m

@torch.no_grad()
def get_action(policy,obs,std):
    m = get_distribution(policy,obs,std)
    act = m.sample().cpu().numpy()
    return act

def reward_to_go(rew):
    rtg = np.zeros_like(rew)
    for i in reversed(range(len(rew))):
        rtg[i] = rtg[i+1] + rew[i] if i < len(rew)-1 else rew[i]
    return rtg

def train(
        env_name = 'BipedalWalker-v3',
        hidden_size = [64,64],
        batch_num = 80000,
        epochs = 100000,
        lr =0.0001,
):
    env = gym.make(env_name)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    policy = network([obs_dim]+hidden_size+[act_dim])
    log_std = Parameter(torch.zeros(act_dim))
    optimizer = Adam(list(policy.parameters())+[log_std],lr=lr)

    for i in range(epochs):
        obs,_ = env.reset()
        batch_obs = []
        batch_act = []
        batch_ret = []
        batch_rew = []
        ep_rew = []

        while True:
            act = get_action(policy,torch.as_tensor(obs,dtype = torch.float32),log_std.exp())
            next_obs,reward,terminated,truncated,_=env.step(act)
            batch_obs.append(obs)
            batch_act.append(act)
            ep_rew.append(reward)
            obs = next_obs
            done = terminated or truncated

            if done:
                batch_ret.extend(reward_to_go(ep_rew).tolist())
                batch_rew.append(sum(ep_rew))
                
                if len(batch_obs) >= batch_num:
                    break

                obs,_ = env.reset() 
                ep_rew = []

        batch_obs = torch.as_tensor(np.asarray(batch_obs),dtype = torch.float32)
        batch_act = torch.as_tensor(np.asarray(batch_act),dtype = torch.float32)
        batch_ret = torch.as_tensor(batch_ret,dtype = torch.float32)

        optimizer.zero_grad()
        logprob = get_distribution(policy,batch_obs,log_std.exp()).log_prob(batch_act).sum(-1)
        loss = - (logprob*batch_ret).sum()/len(batch_rew)
        loss.backward()
        optimizer.step()
        print("epoch",i,"finished, the batch return is :",np.mean(batch_rew))


    env.close()

if __name__ == "__main__":
    train() 

             