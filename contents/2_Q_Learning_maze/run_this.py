"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

此次環境是莫凡自己用Tkinter編寫，未來會直接使用別人寫好的環境。ex: openAI, gym...
"""

# 自己寫的模組(class)
from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100): # 跑100回合
        # initial observation (初始設定為1,1)
        observation = env.reset() # 環境觀測值(我所在位置)

        while True: # 在回合中一直玩
            # fresh env (刷新環境)
            env.render()

            # RL choose action based on observation (基於觀測值[索引]挑選a)
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action) #返回值: 下個狀態、獲得獎勵、是否結束(到黑洞或寶藏)

            # RL learn from this transition (s,a,a,s'以這4個參數學習)
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation (把observation_作為下次的s)
            observation = observation_

            # break while loop when end of this episode (到黑洞或寶藏->結束)
            if done:
                break

    # end of game
    print('game over')
    env.destroy() #刪掉環境(Tkinter編寫規則)

if __name__ == "__main__":
    env = Maze() # 設定環境
    RL = QLearningTable(actions=list(range(env.n_actions))) # RL=某種學習方法

    env.after(100, update) # Tkinter編寫規則
    env.mainloop() # Tkinter編寫規則