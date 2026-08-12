"""
This is partial (modified) code from https://huggingface.co/learn/deep-rl-course/en/unit4/hands-on
"""

import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import device

import gymnasium as gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

env_id = "Pendulum-v1"
env = gym.make(env_id)
eval_env = gym.make(env_id)

s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]

class Policy(nn.Module):
    def __init__(self,s_size,a_size,h_size):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(s_size,h_size)
        self.fc2 = nn.Linear(h_size,a_size)
        self.fc2_ = nn.Linear(h_size,a_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        log_std = self.fc2_(x)
        return mu,log_std

    def act(self,state):
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        mu,log_std = self.forward(state)
        log_std = torch.clamp(log_std,-20,2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu,std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0).detach().cpu().numpy(),log_prob

def reinforce(policy,optimizer,n_training_episodes,max_t,gamma,print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1,n_training_episodes):
        saved_log_probs = []
        rewards = []
        state,obs = env.reset()
        for t in range(max_t):
            action,log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        returns =deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns)>0 else 0
            returns.appendleft(gamma*disc_return_t + rewards[t])
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+eps)

        policy_loss = []

        for log_probs,disc_returns in zip(saved_log_probs,returns):
            policy_loss.append(-log_probs*disc_returns)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))

    return scores

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


cartpole_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 10,
    "max_t": 200,
    "gamma": 0.90,
    "lr": 5e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

scores = reinforce(
    cartpole_policy,
    cartpole_optimizer,
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    100,
)
