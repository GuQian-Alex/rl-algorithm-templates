# Q-Learning

## 1. Core Idea

Q-learning uses a Q-table to estimate the action-value function Q(s, a).

For each state s and action a, the Q-table stores a value that represents how good it is to take action a in state s.

## 2. Key Components

- Q-table
- Learning rate α
- Discount factor γ
- Epsilon-greedy exploration
- Environment interaction
- Q-value update

## 3. Update Rule

目标值：

target = r + γ max_a' Q(s', a')

更新公式：

Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

## 4. Pseudocode

```text
Initialize Q(s, a), for all states s and actions a
for each episode:
    Initialize state s
    while episode is not done:
        Select action a using epsilon-greedy:
            With probability epsilon, select a random action
            Otherwise, select a = argmax_a Q(s, a)
        Take action a
        Observe reward r and next state s'
        Update Q-table:
            Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
        s ← s'
    Decay epsilon
```

## 5. Key Notes

### 1. 如何熟练代码？
不要从头到尾死记硬背，从强化学习的基本伪代码开始，先搭建智能体与环境交互的基本循环，再逐步填充信息。
最基本的强化学习代码就是一个双层的循环结构：
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.learn(state, action, reward, next_state, done)

        state = next_state
所有不同的强化学习算法，都体现在agent.select_action()和agent.learn()两个函数中。后面写其他算法都是替换这两个函数就是了。
所以在后续的算法模块化实现中，可以单独分一个agent.py文件和一个train.py文件

### 2.TD方法的数学来源
TD方法的从原理上比较好理解，用一步的更新来代替原来的Monte Carlo方法的更新，但是这个公式真的能收敛到目标的保证，来自数学上的随机近似理论的严格证明。

### 3.epsilon贪心是必要的吗？可不可以去掉？
不可以。epsilon-greedy保证了对整个地图的广泛探索。如果没有这个机制，会极大降低agent的探索能力，最后也找不到最优解。这是exploit-explore的重要环节。

### 4.Q-Learning的核心公式的拆解
看起来是一行代码，其实里面有4个步骤：
old_value = q[state, action]
next_best_value = np.max(q[new_state])
target = reward + gamma * next_best_value
q[state, action] = old_value + alpha * (target - old_value)

### 5.

