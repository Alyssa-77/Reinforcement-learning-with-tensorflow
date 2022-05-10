"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible (移動速度)

# Global variable
N_STATES = 6   # the length of the 1 dimensional world (總距離)
ACTIONS = ['left', 'right']     # available actions (可選擇的動作)
EPSILON = 0.9   # greedy police (90%選擇最佳動作)
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes (總共只玩13回合)
FRESH_TIME = 0.2    # fresh time for one move (0.3秒走一步，看得比較清楚)


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values (全0的表格)
        columns=actions,    # actions's name (left, right)
    )
    # print(table)    # show table (每個state，每個動作的分數)
    return table


def choose_action(state, q_table):
    # This is how to choose an action (選擇動作)
    state_actions = q_table.iloc[state, :] #選擇q_table的某列(state)
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS) # 10%機率隨機選擇動作 或 table該列皆為0(初始狀態)
    else:   # act greedy
        action_name = state_actions.idxmax()    # 90%機率選擇分數高的動作
    return action_name


def get_env_feedback(S, A): #(env:環境，S_:下一步的狀態)
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated (建立一維環境)
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES): #回合
        step_counter = 0
        S = 0 #初始狀態S=0
        is_terminated = False # 初始 是否終止=F
        update_env(S, episode, step_counter)

        while not is_terminated: # 是否終止
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A] #(Q估計值)
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal (Q真實值=R+γ*Qmax)
            else:
                q_target = R     # next state is terminal (下一回合是終止，沒有Qmax，所以Q真實值=R)
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update (新Q=舊Q+α*誤差值)
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
        # print(q_table)
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
