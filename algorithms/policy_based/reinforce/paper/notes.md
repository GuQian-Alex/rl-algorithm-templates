# REINFORCE

## 1. Core Idea

REINFORCE 是一种 基于策略的强化学习算法。它不直接学习每个状态动作的价值，而是直接学习一个动作策略。
如果某个 episode 中，某个动作之后带来了较高回报，那么以后在类似状态下应该更倾向于选择这个动作；如果带来了较差回报，就降低选择它的概率。

## 2. Key Components

- 随机策略网络 $\pi_\theta(a\mid s)$：输出动作分布，通过采样动作进行探索；连续动作对应概率密度。
- On-policy 轨迹采样：用当前策略收集完整 episode，每条轨迹或每批新轨迹进行一次策略更新。
- Monte Carlo reward-to-go：使用每个时间步之后实际获得的累计奖励，不用价值预测补足未来回报（不 bootstrap）。
- Log-derivative trick：利用 $\nabla_\theta p_\theta = p_\theta\nabla_\theta\log p_\theta$，将期望回报的梯度写成可采样估计的形式，无需对环境或动作采样过程求导。
- 可选 baseline：减去与当前动作无关、更新策略时固定的基准值，以降低梯度估计的方差；基础 REINFORCE 不要求价值网络。

基础 REINFORCE 不需要 Q-network、target network、experience replay 或 epsilon-greedy；探索来自随机策略本身。

## 3. Update Rule

令 $T$ 为轨迹长度，$r_t$ 表示在状态 $s_t$ 执行动作 $a_t$ 后获得的奖励，$\gamma$ 为折扣率。这里采用折扣起始状态目标：

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right]
$$

每个时间步的 reward-to-go 为：

$$
G_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_k
$$

对当前策略采样的一条完整轨迹，构造用于自动求导的代理损失：

$$
L(\theta) = -\sum_{t=0}^{T-1}\gamma^t\,\operatorname{stopgrad}(G_t)\,\log\pi_\theta(a_t\mid s_t)
$$

用梯度下降更新参数，等价于沿该轨迹估计的策略梯度上升：

$$
\theta \leftarrow \theta - \alpha\nabla_\theta L(\theta)
$$

- $\alpha$ 为学习率；上式表示普通梯度下降，也可用 Adam 等优化器处理同一损失的梯度。
- 状态、采样动作和回报在这次更新中视为固定数据，只通过 $\log\pi_\theta$ 对策略参数求导；`stopgrad` 表示不对回报反向传播。
- 使用多条轨迹时，对各条轨迹的损失取平均。该 loss 用于产生策略梯度，其数值本身不是策略表现的评估指标。
- 当 $\gamma=1$ 时，外层 $\gamma^t$ 自然为 1；当 $\gamma<1$ 时，对上述目标不能随意省略它。

参考：[Spinning Up：策略梯度与代理损失](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。

## 4. Pseudocode

1. 初始化随机策略网络和优化器。
2. 用当前策略采样一条或一批完整轨迹，采样期间保持策略不变。
3. 根据实际奖励，逆序计算每个时间步的 reward-to-go。
4. 按前述公式，用回报加权动作的 log probability，构造策略损失。
5. 清空梯度、反向传播，更新策略一次。
6. 丢弃旧轨迹，用更新后的策略重新采样，重复上述过程。

## 5. Key Notes

### 1. 为什么要把所有时间步的loss都加和？
Reinforce使用一整条轨迹来更新参数，这条轨迹的概率是所有时间步动作概率的成绩，对log_prob来说就是求和。
其实也可以用平均值，二者只差了一个常数系数，本质上一样，只是梯度尺度不同

### 2. 这个算法中的计算图是怎么样的？
Reinforce中有两个计算图，分别是
1）策略网络本身的计算图：state -> fc1 -> relu -> fc2 -> softmax -> action probability；
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

### 12. reinforce核心解决了一个什么问题？
强化学习的典型目标是最大化J(θ)，最直接的想法是对J直接求导，θ→a0​→s1​→a1​→s2​→⋯→R.
但是存在两大困难，动作采样不可导和环境不可导。
直接求导通过改变 action 的数值来提高 R；REINFORCE 通过改变 action 的概率分布，使高回报 action 更可能被采样。
两者都优化期望回报，直接求导调整样本路径，REINFORCE 调整产生这些路径的概率。

### 13.REINFORCE有非常多的改进和工程细节
一个重要的点是action的截断问题，这个在黑色笔记本上进行了总结。
一个重要的是reward to go， learned baseline，advantage function的等的改进，learning baseline实质上引入了可以学习的V网络，有点actor critic的结构了。这些都是针对值函数的改进。
还有一种是针对策略函数的改进，例如KL置信域、clip截断等，这条路上衍生出了TRPO、PPO等


### 14.REINFORCE的最大的问题是方差太高，但是这个问题可以用更多的数据和batch来解决

更准确地说，大 batch 能缓解而非彻底解决高方差：在固定策略下，对 $B$ 条独立同分布轨迹的梯度求平均，其方差为单轨迹梯度方差的 $1/B$。
仅增加总训练回合数、仍逐轨迹更新，并不会直接降低每次更新的方差。
增大 batch 会增加每次更新的采样成本，在总样本数固定时还会减少更新次数；通常需要配合 baseline、合理探索和学习率设置，不能保证收敛。
增大batch也会增加训练收尾时候的稳定性，小batch很容易在长训练的末尾发生性能的衰退

### 15.Critic 除了作为 MC baseline，还可以做什么？

Critic 的本质是“估计长期价值”，作为 MC baseline 只是它的一种用法。

1. 预测未来回报：用于 TD bootstrap

   $$
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
   $$

2. 构造更灵活的 advantage：n-step 和 GAE

3. 直接指导动作优化：学习 $Q(s,a)$

   $$
   L_{\text{actor}} = -\mathbb{E}_s\left[Q_\phi(s,\mu_\theta(s))\right]
   $$
