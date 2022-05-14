"""
Reinforcement learning quic example.

state: delay, bandwidth, throughput, loss
reward: delay, throughput, loss

This script is the environment part of this example. The RL is in RL_brain.py.
"""

import numpy as np
import time

# reward weights
A = 0.4 
B = 0.4
C = 0.2


# "run.py" import this class
class Quic(object):
    def __init__(self):
        super(Quic, self).__init__()
        self.action_space = [4000, 8000, 10000, 20000, 40000, 80000, 100000, 200000, 400000, 800000, 1000000] # stream 16KB~16MB (11 sizes)
        self.n_actions = len(self.action_space)
        self.title('quic')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_quic()

    def _build_quic(self):
        self.delay = 0
        self.bw = 0
        self.tput = 0
        self.loss = 0

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self): # return 觀測值 (state)
        self.update()   # 回合結束，更新
        time.sleep(0.5)
        # todo 回到初始狀態 f=10000, c=1000000
        # 測得d,b,p,o
        self.delay = 0
        self.bw = 0
        self.tput = 0
        self.loss = 0
        return self

    def step(self, action): # return s', done, r
        s = [self.delay, self.bw, self.tput, self.loss] # d b p o
        # todo 依照action跑quic

        # todo 取得新s，放入s_
        s_ = [self.delay, self.bw, self.tput, self.loss]  # next state

        # reward function, done
        if s_[2] == 0:      # todo 何時done?
            reward = 0
            done = True
            s_ = 'terminal'
        else:
            reward = A*s_[0]+B*s_[2]-C*s_[3]  # todo reward設計，Z標準化?
            done = False

        return s_, reward, done

    def render(self): # todo not sure
        time.sleep(0.1)
        self.update()

# todo not sure
def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Quic()
    env.after(100, update)
    env.mainloop()