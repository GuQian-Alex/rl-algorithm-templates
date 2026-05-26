# REINFORCE

## 1. Core Idea

REINFORCE 是一种 基于策略的强化学习算法。它不直接学习每个状态动作的价值，而是直接学习一个动作策略。
如果某个 episode 中，某个动作之后带来了较高回报，那么以后在类似状态下应该更倾向于选择这个动作；如果带来了较差回报，就降低选择它的概率。

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

## 5. Key Notes

### 1. 为什么要把所有时间步的loss都加和？
Reinforce使用一整条轨迹来更新参数，这条轨迹的概率是所有时间步动作概率的成绩，对log_prob来说就是求和。
其实也可以用平均值，二者只差了一个常数系数，本质上一样，只是梯度尺度不同

### 2. 这个算法中的计算图是怎么样的？
Reinforce中有两个计算图，分别是1）策略网络本身的计算图：state -> fc1 -> relu -> fc2 -> softmax -> action probability；
2）REINFORCE 更新时的计算图：log_prob -> policy_loss -> backward -> 更新 policy 参数。
reward、return、env.step() 不在 PyTorch 计算图里；真正参与反向传播的是 log_prob 这条链路。
假设一个episode走了4步，那么其计算图为：
s0 ─> policy θ ─> probs0 ─> log_prob0 ─┐
                                        │
s1 ─> policy θ ─> probs1 ─> log_prob1 ─┤
                                        │
s2 ─> policy θ ─> probs2 ─> log_prob2 ─┤──> policy_loss ─> backward() ─> θ.grad
                                        │
s3 ─> policy θ ─> probs3 ─> log_prob3 ─┘

rewards ─> returns ────────────────────┘    return只作为权重参与计算，没有梯度
反向传播时，所有计算图的分支的梯度都会加起来，也是体现了1中的loss加和

### 3. optimizer.zero_grad()是什么意思？
optimizer.zero_grad()的作用是清空上一轮的梯度，防止混淆

### 4. 这种policy-based方法为什么不直接用深度学习的思路，用return来直接训练策略网络参数？
因为有两个地方不可导：1. action 是离散采样结果，不可导 2. env.step 是外部环境，不在 PyTorch 计算图里。REINFORCE 的聪明之处是绕开这条路， 不对采样动作本身求导，而是对“采样到这个动作的概率”求导。
如果reward 或环境动力学本身是可导的，真的可以把 reward 反向传播回神经网络参数。
一个统计学里很常见的视角：参数不一定改变样本值本身，它可以改变样本出现的概率。reinforce就是做了这样一个事情


### 5. 与DQN相比，Reinforce算法是如何把深度学习与强化学习结合起来的？
Reinforce并不是直接找两个目标值的差作为loss，去反向传播更新自己的网络。
表面上Reinforce是在做普通神经网络 loss 最小化；本质上，因为 loss 被设计成了公式中的样子， 它实现的正是 REINFORCE 的策略梯度更新。
最后的结果是，pytorch在做loss的最小化时，实际上在做Reinforce要求的期望最大化。
这个 loss 不是像监督学习里的 MSE 或交叉熵那样，直接表达“预测值和标签的误差”。
它更像是一个 为了让自动求导产生正确策略梯度而构造出来的代理目标，也叫 surrogate loss。

### 6. 神经网络前向传播时也经过了softamx和relu，在backward的时候也会求这些函数的梯度吗？
会。只要这些操作参与了从 policy_loss 到网络参数的计算路径，PyTorch 在 backward() 时就会自动对它们求梯度，包括 ReLU、Softmax、Linear、log_prob 等。
1. ReLU 会让小于 0 的位置梯度变成 0。
2. Softmax 会让动作概率之间产生联动梯度。

### 7. reinforce算法一开始就是为了绕开不可直接求导问题而想出来的吗？
是的，REINFORCE 最初的动机基本就是：在不能直接对实际奖励路径求导的情况下，仍然得到“期望奖励对策略参数的梯度”。
它的理论出发点本来就是：我们想最大化期望奖励，但是采样动作、随机神经元、环境反馈这些东西会让普通反向传播断掉；有没有一种统计方法，能直接估计期望奖励的梯度？

### 8. likelihood-ratio trick
REINFORCE 通过 log-derivative trick，把“最大化期望回报”的梯度，改写成一个可以通过采样估计的期望。
每次采样一条轨迹后，用这条轨迹里的 log_prob(action) * return 构造一个随机梯度估计。
这个随机梯度的期望，等于我们真正想要的目标函数梯度。
因此，虽然单次更新有噪声，但平均来说，它是在朝着提高期望回报的方向更新神经网络。

### 9.为什么期望公式中的G_t可以替换成Q？
通过条件期望可以证明期望中的G_t可以替换成Q，Spinningup中有严格证明

### 10.Reinforce的创新点在哪里？
REINFORCE 的新意主要在于：在随机策略、随机采样、环境不可导的情况下，给出了一个可以用样本估计的策略梯度更新方法。
“最大化期望累计回报”是强化学习和最优控制里更早、更基础的目标设定。

### 11.后面的TD方法修改了什么？
理论目标仍然是对所有时间步的 actor loss 项求和；TD 方法只是让每个时间步的目标可以即时计算，因此实现上可以逐步更新，而不必等整条轨迹结束后再统一求和。
根据策略梯度的公式，所有actor-critic，无论TD与否，目标函数的梯度对应的期望是对整个轨迹的log_prob求和的期望，所有的算法都要这么做。
TD和Monte Carlo的区别只在于更新的步频不一样而已。