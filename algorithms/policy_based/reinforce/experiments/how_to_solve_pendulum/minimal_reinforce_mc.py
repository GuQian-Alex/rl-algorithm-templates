"""Pendulum：极简 REINFORCE + learned Monte Carlo baseline（efficient_30k）。

原配置在 20 个新 seed 中成功 19 次；本文件保留其默认训练计算和随机流。
CPU，30,000 episodes，batch=16；只保留训练与打印，不评估、不保存。
依赖 gymnasium、numpy、torch；运行：python minimal_reinforce_mc.py
短程检查：train(episodes=18)，返回 (actor, critic)，不代表收敛实验。
修改训练长度或 batch 属于自定义实验，不沿用原配置的验收结果。
"""

import math

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

HORIZON = 200
ACTOR_LR = 0.003


def scale_observation(states):
    return states / states.new_tensor([1.0, 1.0, 8.0])


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor 保留默认初始化，不使用 Critic 的正交初始化。
        self.mean = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def distribution(self, states):
        mean = self.mean(scale_observation(states))
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def action(self, states, deterministic=False):
        dist = self.distribution(states)
        raw = dist.mean if deterministic else dist.sample()
        return 2.0 * raw.tanh(), raw

    def log_prob(self, states, raw):
        # 保存 tanh 前的采样值，稳定计算动作变换的 Jacobian。
        jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return (self.distribution(states).log_prob(raw) - jacobian - math.log(2)).sum(-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for inputs in (4, 64):
            layer = nn.Linear(inputs, 64)
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.Tanh()])
        output = nn.Linear(64, 1)
        nn.init.orthogonal_(output.weight, 1.0)
        nn.init.zeros_(output.bias)
        self.net = nn.Sequential(*layers, output)

    def forward(self, states, times):
        inputs = torch.cat([scale_observation(states), (times / HORIZON).unsqueeze(-1)], -1)
        return self.net(inputs).squeeze(-1)


def reward_to_go(rewards):
    # gamma=1；完整 MC 回报，末端为 0，不从 Critic bootstrap。
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        running = rewards[t] + running
        returns[t] = running
    return returns


@torch.no_grad()
def collect(actor, critic, envs, reset_seed):
    states = np.stack([env.reset(seed=reset_seed + i)[0] for i, env in enumerate(envs)])
    data = {key: [] for key in ("states", "raw", "rewards", "values")}
    for t in range(HORIZON):
        x, times = torch.as_tensor(states), torch.full((len(envs),), t)
        actions, raw = actor.action(x)
        transitions = [env.step(action) for env, action in zip(envs, actions.numpy())]
        nxt, rewards, terminated, truncated, _ = zip(*transitions)
        if any((end or limit) != (t == HORIZON - 1) for end, limit in zip(terminated, truncated)):
            raise RuntimeError("本模板要求 Pendulum-v1 的完整 200-step 轨迹")
        data["states"].append(x)
        data["raw"].append(raw)
        data["rewards"].append(torch.tensor(rewards, dtype=torch.float32))
        data["values"].append(critic(x, times))
        states = np.stack(nxt)
    data = {key: torch.stack(value) for key, value in data.items()}
    data["times"] = torch.arange(HORIZON)[:, None].expand(-1, len(envs))
    data["targets"] = reward_to_go(data["rewards"] * 0.001)
    # 使用更新前的 baseline；不居中、不标准化，且不保留梯度。
    data["advantages"] = data["targets"] - data["values"]
    return data


def step(loss, network, optimizer):
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 0.5, error_if_nonfinite=True)
    optimizer.step()


def update(actor, critic, actor_opt, critic_opt, data, shuffle_seed):
    states, raw = data["states"].reshape(-1, 3), data["raw"].reshape(-1, 1)
    times, targets, advantages = [data[key].flatten() for key in ("times", "targets", "advantages")]
    # 每批只做一次全量 Actor 更新；gamma=1，所以外层 gamma**t=1。
    loss = -(actor.log_prob(states, raw) * advantages).mean()
    step(loss, actor, actor_opt)
    # 再用固定 MC target 拟合 Critic；不重复更新 Actor。
    generator = torch.Generator().manual_seed(shuffle_seed)
    for _ in range(10):
        for indices in torch.randperm(len(states), generator=generator).split(256):
            loss = F.mse_loss(critic(states[indices], times[indices]), targets[indices])
            step(loss, critic, critic_opt)


def training_seed_base(seed):
    # 保留已验收实现的 reset 序列，避免精简时换掉实验随机流。
    development = [2101, 2203, 2309]
    if seed in development:
        return 1000000000 + development.index(seed) * 1000000
    if 9001 <= seed <= 9020:
        return 2000000000 + (seed - 9001) * 1000000
    return 3000000000 + seed * 1000000


def train(episodes=30_000, batch_episodes=16, seed=9001):
    if episodes <= 0 or batch_episodes <= 0 or not 0 <= seed < 2**32:
        raise ValueError("episodes、batch_episodes 必须为正，seed 必须在 [0, 2**32) 内")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    actor = Actor()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 50000)
        critic = Critic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=0.001)
    envs = [gym.make("Pendulum-v1") for _ in range(min(batch_episodes, episodes))]
    try:
        for batch_index, episode in enumerate(range(0, episodes, batch_episodes)):
            width = min(batch_episodes, episodes - episode)
            data = collect(actor, critic, envs[:width], training_seed_base(seed) + episode)
            actor_opt.param_groups[0]["lr"] = ACTOR_LR * (1 - 0.9 * (episode / episodes))
            update(actor, critic, actor_opt, critic_opt, data,
                   shuffle_seed=1000000000 + seed * 100000 + batch_index)
            print(f"episode {episode + width:5d} | train return {data['rewards'].sum(0).mean().item():.2f}")
    finally:
        for env in envs:
            env.close()
    return actor, critic


if __name__ == "__main__":
    actor, critic = train()
