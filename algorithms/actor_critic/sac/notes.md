# DQN

## 1. Core Idea

DQN uses a neural network to approximate the action-value function Q(s, a).

## 2. Key Components

- Q-network
- Target network
- Experience replay
- Epsilon-greedy exploration

## 3. Update Rule

目标值：

y = r + γ max_a' Q_target(s', a')

损失函数：

L = (Q(s, a) - y)^2

## 4. Pseudocode

1. Initialize replay buffer
2. Initialize Q-network and target network
3. Interact with environment
4. Store transitions
5. Sample mini-batch
6. Update Q-network
7. Periodically update target network

## 5. Supported Environments

- CartPole-v1
- LunarLander-v2

## 6. Run


### 12.正常采样和重参数化采样的区别？为什么reinforce中可以用正常采样，而SAC中需要重参数化采样？

reinforce中，action不在计算图之内，更新公式中不需要对action求梯度。
sac中，由于Q的存在，action必须存在于计算图之内，更新公式需要对action求梯度，所以只能用重参数化采样。

需要仔细梳理一下两个算法中的计算图，才能搞清楚。为什么两个算法都用到动作采样，但是一个的不在计算图，一个在计算图。