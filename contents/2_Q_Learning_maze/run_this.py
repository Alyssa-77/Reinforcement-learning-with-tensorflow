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
from maze_env import Maze           # 自己的環境
from RL_brain import QLearningTable # RL


def update():
    for episode in range(100): # 跑100回合
        observation = env.reset() # 環境觀測值=我的位置 (reset初始設定為1,1)

        while True:         # 在回合中一直玩
            env.render()    # 刷新環境

            action = RL.choose_action(str(observation))     # choose_action：基於觀測值state挑選a (state是Q table的索引)
            observation_, reward, done = env.step(action)   # step 返回值: 下個狀態、獲得獎勵、是否結束(到黑洞或寶藏)
            RL.learn(str(observation), action, reward, str(observation_)) # learn：s,a,a,s' 以這4個參數學習
            observation = observation_ # 把observation_作為下次的s

            if done:    # 到黑洞或寶藏->結束這回合
                break

    print('game over')  # end of game
    env.destroy()       # 刪掉環境(Tkinter編寫規則)



if __name__ == "__main__":
    env = Maze() # 設定環境
    RL = QLearningTable(actions=list(range(env.n_actions))) # RL=學習方法 (action=n_actions=['u', 'd', 'l', 'r'])

    env.after(100, update) # Tkinter編寫規則
    env.mainloop() # Tkinter編寫規則