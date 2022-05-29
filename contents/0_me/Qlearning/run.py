"""
Reinforcement learning quic example.

state: delay, bandwidth, throughput, loss
reward: delay, throughput, loss

This script is the main part which controls the update method of this example.
The RL is in myRL.py.
"""

# 自己寫的模組(class)
from quic_env import Quic           # 自己的環境
from RL import QLearningTable       # RL

def run():
    print("RUN")
    for episode in range(2):        # 跑x回合 todo
        observation = env.reset()   # 環境觀測值 (reset初始設定)

        while True:         # 在回合中一直玩
            # env.render()    # 刷新環境(每一小步，還沒到黑洞或寶藏之前)

            action = RL.choose_action(str(observation))     # 基於觀測值observation挑選a (observation是Q table的索引)
            observation_, reward, done = env.step(action)   # 給了動作後，返回: 下個狀態、獲得獎勵、是否結束(到黑洞或寶藏)
            RL.learn(str(observation), action, reward, str(observation_)) # learn：s,a,a,s' 以這4個參數學習
            observation = observation_ # 把observation_作為下次的s

            if done:    # 到黑洞或寶藏->結束這回合
                # print(RL.q_table, "\n")
                print("DONE: ",episode,"\n")
                break

    print('game over')  # end of game



if __name__ == "__main__":
    env = Quic() # 設定環境
    RL = QLearningTable(actions=list(range(env.n_actions))) # RL=學習方法 
    # env.after(100, run) # Tkinter編寫規則 # 時間間隔之後，執行指定的函數
    run()


    # n_actions=[4000, 8000, 10000, 20000, 40000, 80000, 100000, 200000, 400000, 800000, 1000000] (16KB~16MB, 11 sizes)