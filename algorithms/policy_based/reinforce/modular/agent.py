import torch
import torch.nn as nn
from torch.distributions import Categorical,Normal
from torch.nn import Parameter




class Agent(nn.Module):

    def __init__(self,sizes,act_type,activation=nn.Tanh):
        super().__init__()
        self.sizes = sizes
        self.act_type = act_type
        if act_type not in ("continuous", "discrete"):
          raise ValueError(f"Unsupported act_type: {act_type}")
        self.activation = activation
        self.creat_net()


    def creat_net(self):
        layers = []
        for i in range(len(self.sizes)-1):
            act_func = self.activation if i <len(self.sizes)-2 else nn.Identity
            layers += [nn.Linear(self.sizes[i],self.sizes[i+1]),act_func()]
        self.policy = nn.Sequential(*layers)

        if self.act_type == 'continuous':
            self.log_std = Parameter(torch.zeros(self.sizes[-1]))

    def get_distribution(self,obs):
        if self.act_type == 'continuous':
            mu = self.policy(obs)
            m = Normal(mu,self.log_std.exp())

        if self.act_type == 'discrete':
            logits = self.policy(obs)
            m = Categorical(logits=logits) 

        return m

    def get_logprob(self,obs,act):
        m = self.get_distribution(obs)
        logprob = m.log_prob(act)

        if self.act_type == "continuous":
           logprob=logprob.sum(dim=-1)
        return logprob

    @torch.no_grad()
    def get_action(self,obs):
        m = self.get_distribution(obs)
        action = m.sample()
        return action


    
    