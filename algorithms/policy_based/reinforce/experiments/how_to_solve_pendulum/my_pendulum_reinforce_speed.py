"""Pendulum 大 batch Monte Carlo REINFORCE：从 2026-11-10 实验源码精简。

依赖：numpy==2.1.3 torch==2.5.1 gymnasium==1.1.1（原实验版本）。
运行：python pendulum_reinforce_minimal.py
默认 50,000 条完整 episode / batch，300 次更新，共 30 亿环境步。
仅检查能否运行时：import 本文件后调用 train(batch_episodes=5, updates=2)。

保留原始算法及向量化环境；删除日志、评估、保存、断点续训和实验管理。
CPU 运行，训练函数返回策略。没有 critic、GAE、PPO、熵奖励或梯度裁剪。
"""

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

BATCH_EPISODES = 50_000
UPDATES = 300
LEARNING_RATE = 0.001
GAMMA = 0.97
HORIZON = 200
MICROBATCH = 8192
SEED = 41000


class GaussianPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里的mean就是神经网络，是用来计算mean的神经网络命名为mean
        self.mean = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        # 原实验初始 std=1；均值、log_std 都不额外裁剪。
        self.log_std = nn.Parameter(torch.zeros(1))

    def log_prob(self, observations, raw_actions):
        distribution = Normal(self.mean(observations), self.log_std.exp())
        return distribution.log_prob(raw_actions).sum(-1)


class ArrayPendulum:
    """批量 Pendulum-v1 动力学；Gymnasium 负责生成逐 episode 初始状态。"""

    def __init__(self, batch):
        self.batch = batch
        self.reset_env = gym.make("Pendulum-v1")

    def observation(self):
        # pendulum的核心参数有两个，theta velocity，根据这两个核心state就可以计算的得出observation
        theta, velocity = self.state[:, 0], self.state[:, 1]
        return np.stack((np.cos(theta), np.sin(theta), velocity), axis=1).astype(np.float32)

    def reset(self, seeds):
        self.state = np.empty((self.batch, 2), dtype=np.float64)
        for i, seed in enumerate(seeds):
            self.reset_env.reset(seed=int(seed))
            #  穿过所有包装起，取得最底层的原始环境对象
            self.state[i] = self.reset_env.unwrapped.state
        return self.observation()

    def step(self, raw_actions):
        theta, velocity = self.state[:, 0], self.state[:, 1]
        torque = np.clip(raw_actions[:, 0], -2.0, 2.0)
        costs = (
            ((theta + np.pi) % (2 * np.pi) - np.pi) ** 2
            + 0.1 * velocity**2 + 0.001 * torque**2
        )
        new_velocity = velocity + (
            3 * 10.0 / (2 * 1.0) * np.sin(theta)
            + 3.0 / (1.0 * 1.0**2) * torque
        ) * 0.05
        new_velocity = np.clip(new_velocity, -8, 8)
        new_theta = theta + new_velocity * 0.05
        self.state = np.stack((new_theta, new_velocity), axis=1)
        return self.observation(), -costs

    def close(self):
        self.reset_env.close()


@torch.no_grad()
def collect(policy, env, seeds, noise):
    """整批采样期间策略不更新；每条轨迹恰好 200 步。"""
    observations = np.empty((env.batch, HORIZON, 3), np.float32)
    actions = np.empty((env.batch, HORIZON, 1), np.float32)
    rewards = np.empty((env.batch, HORIZON), np.float64)
    obs = env.reset(seeds)
    for t in range(HORIZON):
        observations[:, t] = obs
        raw = policy.mean(torch.from_numpy(obs)) + policy.log_std.exp() * torch.from_numpy(noise[:, t].copy())
        actions[:, t] = raw.numpy()
        obs, rewards[:, t] = env.step(actions[:, t])
    return observations, actions, rewards


def normalized_returns(rewards, gamma=GAMMA):
    """完整有限时域 MC reward-to-go；末尾为 0，不 bootstrap。"""
    running = np.zeros(rewards.shape[0], np.float64)
    returns = np.empty(rewards.shape, np.float32)
    for t in range(HORIZON - 1, -1, -1):
        running = rewards[:, t].astype(np.float64) + float(gamma) * running
        returns[:, t] = running
    flat = torch.from_numpy(returns.reshape(-1))
    # 对整个 batch 的所有时间步统一标准化，而不是每条轨迹分别标准化。
    return (flat - flat.mean()) / (flat.std(correction=1) + float(np.finfo(np.float32).eps))


def update(policy, optimizer, observations, actions, rewards, microbatch=MICROBATCH):
    weights = normalized_returns(rewards)
    obs = torch.from_numpy(observations.reshape(-1, 3))
    act = torch.from_numpy(actions.reshape(-1, 1))
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(weights), microbatch):
        end = min(len(weights), start + microbatch)
        # 保留原实验的 sum，而非 mean；不额外乘 gamma**t。
        # 使用未裁剪的采样动作计算 log_prob，环境内才裁剪执行力矩。
        loss = -(policy.log_prob(obs[start:end], act[start:end]) * weights[start:end]).sum()
        loss.backward()
    # microbatch 仅节省反向传播内存，不是多次优化、也不重用数据。
    optimizer.step()


def derived_seed(seed, role):
    return int(np.random.SeedSequence([2026111001, seed, role]).generate_state(1)[0])


def train(batch_episodes=BATCH_EPISODES, updates=UPDATES, seed=SEED):
    if batch_episodes <= 0 or updates <= 0:
        raise ValueError("batch_episodes and updates must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derived_seed(seed, 1))
        policy = GaussianPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=1e-8)
    reset_rng = np.random.default_rng(derived_seed(seed, 3))
    noise_rng = np.random.default_rng(derived_seed(seed, 4))
    env = ArrayPendulum(batch_episodes)
    try:
        for _ in range(updates):
            seeds = reset_rng.integers(0, 2**32, size=batch_episodes, dtype=np.uint32)
            noise = noise_rng.standard_normal((batch_episodes, HORIZON, 1)).astype(np.float32)
            observations, actions, rewards = collect(policy, env, seeds, noise)
            del noise
            update(policy, optimizer, observations, actions, rewards)
            del observations, actions, rewards
    finally:
        env.close()
    return policy


if __name__ == "__main__":
    policy = train()
