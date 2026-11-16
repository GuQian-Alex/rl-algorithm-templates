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

def reward_to_go(rew):
    rtg = torch.zeros_like(rew)
    for i in reversed(range(len(rew))):
        rtg[i] = rtg[i+1] + rew[i] if i < len(rew)-1 else rew[i]
    return rtg

def train(
        env_name = 'BipedalWalker-v3',
        hidden_size = [64,64],
        batch_num = 20,
        epochs = 1000,
        lr =0.001,
):
    