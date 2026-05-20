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

```bash
python train.py