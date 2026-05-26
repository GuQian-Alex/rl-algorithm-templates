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

y = r + γ max_a'[Q_target(s', a')]

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

## 5. Key Notes

### 1. 相比于Q-learning有什么变化？
1）引入了神经网络，加入了深度学习的相关内容，例如Loss、Optimizer、backward等内容
2）加入了Replay buffer、Target network两个新工具

### 2. Q-network的输入输出是什么？
输入是State，输出是该State下的所有的Q值，而不是具体哪个动作

### 3. Loss是怎么构建的？
Loss用被选中执行的action的Q值跟Bellman target的MSE构建。
虽然表面上最后的Loss只是两个标量值的产生，而不是由网络输出的Q值向量产生，但是还是要用Q向量，因为参数共享，需要反向传播
在Q-learning中只修改单个数值，而非参数共享，牵一发动全身。

### 4.target network的作用是什么？
target的作用是根据输入的下一状态，计算Q(s',a)，用于计算Bellman target
在JohnnyCode的DQN代码中，用 target_dqn(state) 来生成其他动作的目标值，而不是用 current_q.clone().detach()。
如果 target_dqn 和 policy_dqn 参数完全一样，那么其他动作值就和 current_q 一样；
但如果两个网络已经不同步一段时间，那么其他动作的 Q 值可能不完全一样。
可能会造成Loss数值的偏差。
标准做法还是current_q.clone().detach()

### 5.detach（）函数是做什么的？
detach的作用是切断计算图，阻止反向传播。另一种类似的功能的代码是torch.no_grad()：
with torch.no_grad():
    next_q_values = target_net(next_states)
    max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
    target = rewards + gamma * max_next_q * (1 - dones)
意思是在这个代码块里，PyTorch 不记录计算图。
PyTorch 有自动求导系统 autograd，只要某个 Tensor 设置了requires_grad=True，PyTorch 就会记录它后面参与过的计算。
DQN中不希望通过 target 这边反向传播去更新 target network，所以给target切断计算图

### 6. pytorch自动计算出来的梯度存在那里？
grad 就存在每一个可训练参数对象自己的 .grad 属性里。
神经网络里的参数不是普通数字，而是 torch.nn.Parameter 对象；
每个 Parameter 都是一个 Tensor，并且通常默认 requires_grad=True。
net.parameters()自动返回网络参数的迭代器，使得optimizer进行优化更新。
pytorch数据默认不求梯度，参数默认求梯度。

### 7.
