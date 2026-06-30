import gymnasium as gym
import numpy as np


def run():

    env = gym.make("FrozenLake-v1",map_name="8x8",is_slippery=True,render_mode="human")

    episodes = 10
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()

    for i in range(episodes):
    #    Q =
    #    reward =
    #    terminated = False
    #    truncated = False
    #    obs = env.reset()
    #    while not terminated and truncated:
    #            action = Q(obs)
    #            obs,reward,terminated,truncated,_=env.step(action)
    #            Q.update()

        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if  rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(Q[state,:])
            # action = env.action_space.sample()
            new_state,reward, terminated,truncated,_=env.step(action)
            reward = float(reward)

            Q[state,action] = Q[state,action] + \
                  learning_rate * \
                  (reward + discount_factor*np.max(Q[new_state,:]) - Q[state,action])


            state = new_state
        epsilon = max(epsilon - epsilon_decay_rate, 0)


    env.close()

if __name__ == "__main__":
    run()

