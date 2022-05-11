'''
Using:
python 3.7.4  (Windows)
python 3.8.10 (Linux)
tensorflow 1.15
'''

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2) # 隨機種子
tf.set_random_seed(2)  # reproducible 可重現的

OUTPUT_GRAPH = True # 輸出 tensorboard 文件
# DISPLAY_REWARD_THRESHOLD = 200  # 當 回合總reward > 200 時，顯示模擬窗口
# RENDER = False  # 在螢幕上顯示模擬畫面。(只會回應出呼叫那一刻的畫面給你，要持續出現，需要寫迴圈)

# Superparameters 定義參數
MAX_EPISODE = 100      # episode次數
MAX_EP_STEPS = 1000     # 1個episode最多可以有幾個step
GAMMA = 0.9     # reward  r discount in TD error
LR_A = 0.001    # actor的 learning rate 
LR_C = 0.01     # critic的 learning rate

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped # 取消限制

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

'''
======================================= Class =======================================
'''
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # placeholde佔位符
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(       # dense=FC
                inputs=self.s,
                units=20,               # 隱藏層的unit數量
                activation=tf.nn.relu,  # RELU
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights，高斯分布(平均值=0 , 標準差=1)。NN權重的推薦初始值。
                bias_initializer=tf.constant_initializer(0.1),  # biases，初始值(value)。生成一個初始值為常數的tensor物件。
                name='l1'
            )

            self.acts_prob = tf.layers.dense(   # dense=FC
                inputs=l1,              
                units=n_actions,                # output units
                activation=tf.nn.softmax,       # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'): #預計價值
            log_prob = tf.log(self.acts_prob[0, self.a]) #log(機率)
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # (log機率)*(TD_error)，機率要增還減。

        with tf.variable_scope('train'): #最大化獎勵=最小化(-V)
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # 想最大化預計價值(exp_v) = 最小化(-exp_v)

    def learn(self, s, a, td): #透過s,a,td 學習該加大幅度或減小
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s): #依據機率選擇動作
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(       # FC
                inputs=self.s,
                units=20,               # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'): #TD_error當作loss，進行誤差的反向傳遞
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_): #td_error = V(下個狀態) - V(上個狀態)  價值相減
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error

'''
======================================= Main =======================================
'''
sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)      # critic應該要學得比actor快，所以LR_C > LR_A

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH: # tensorboard 分析圖
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE): #遊戲回合
    s = env.reset()
    t = 0
    track_r = []
    while True: # 每一步
        # if RENDER: env.render()

        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)

        if done: r = -20
        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]  #critic給出評分：td_error
        actor.learn(s, a, td_error) # true_gradient = grad[logPi(s,a) * td_error] #actor用td_error學習

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # 顯示圖
            print("episode:", i_episode, "  reward:", int(running_reward)) # todo 存成矩陣，印成圖
            break

