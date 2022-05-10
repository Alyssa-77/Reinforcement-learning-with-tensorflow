"""
Policy Gradient, Reinforcement Learning.
The cart pole example (遊戲1)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # 當 回合總reward > 400 時，顯示模擬窗口
RENDER = False  # 在螢幕上顯示模擬畫面會拖慢速度, 等學得差不多了再顯示模擬

env = gym.make('CartPole-v0') #CartPole 這個模擬
env.seed(1) # 普通Policy gradient, 回合的 variance 較大, 選一個好點的隨機種子
env = env.unwrapped # 取消限制

print(env.action_space)         # 顯示可用 action
print(env.observation_space)    # 顯示可用 state 的 observation
print(env.observation_space.high)   # 顯示 observation 最高值
print(env.observation_space.low)    # 顯示 observation 最低值

#定義
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,      # gamma
    # output_graph=True,    # 輸出 tensorboard 文件
)

for i_episode in range(3000): # 一回合中可以做3000步，做完再開始學 (回合更新)
    observation = env.reset()

    while True:
        if RENDER: env.render() # 視覺化呈現，只會回應出呼叫那一刻的畫面給你，要持續出現，需要寫個迴圈。

        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward)    # 儲存這一回合的 transition

        if done: #這回合做完，開始學
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # reward>400，RENDER = True，顯示模擬
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn() # 學習

            if i_episode == 0:
                plt.plot(vt)    # plot 這回合的vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
