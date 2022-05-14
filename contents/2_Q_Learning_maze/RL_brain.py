"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/
"""

import numpy as np
import pandas as pd

#  "run_this.py" import this class
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9): #建立一個空Qtable 
        self.actions = actions      # a list
        self.lr = learning_rate     # learning_rate = alpha
        self.gamma = reward_decay   # reward_decay = gamma
        self.epsilon = e_greedy     # e_greedy = ε epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):       #state是Q table的索引
        self.check_state_exist(observation)     #檢查observation索引是否存在

        if np.random.uniform() < self.epsilon:  #若隨機數<ε(0.9)，90%機率使用最優解a
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:                                   #若隨機數>ε(0.9)，10%機率隨機選擇a
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_) # 也要檢查s_是否在Qtable中
        q_predict = self.q_table.loc[s, a] # Q估計值
        if s_ != 'terminal':    
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()   # not done
        else:
            q_target = r  # done, next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)      # update

    def check_state_exist(self, state):             # 不知道Q table大小，檢查此索引(state)是否存在
        if state not in self.q_table.index:         # 若從來沒經歷過此s，新增到Q table
            self.q_table = self.q_table.append(     # append new state to q table
                pd.Series( #全0
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )