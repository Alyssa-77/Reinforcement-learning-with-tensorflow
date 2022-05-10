"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.
The Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

# [連續動作] 立桿子遊戲

import multiprocessing
import threading
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True #是否輸出tf圖
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count() #worker數量=CPU數量
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


##################### ActorCrtic Net ########################
class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:   #scope判斷是否為global network
            with tf.variable_scope(scope): #創建大腦
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:] #提取a、c的參數
        else:   # local net, calculate losses
            with tf.variable_scope(scope): #創建分身
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error') #計算TD error
                with tf.name_scope('c_loss'): #critic loss：減小error
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'): #actor用到的機率二次分布
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                normal_dist = tf.distributions.Normal(mu, sigma) #生成一個normal distribution

                with tf.name_scope('a_loss'): #actor loss
                    log_prob = normal_dist.log_prob(self.a_his) #distribution的log probability
                    exp_v = log_prob * tf.stop_gradient(td) #期望的動作值
                    entropy = normal_dist.entropy()  # encourage exploration 增加隨機度
                    self.exp_v = ENTROPY_BETA * entropy + exp_v #若太肯定要選某動作，就增加隨機性，讓你不那麼肯定
                    self.a_loss = tf.reduce_mean(-self.exp_v) #actor loss：反向的exp value (因tf只有最小化，所以加負號)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1]) #選動作：根據normal_distribution採樣
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params) #a的梯度
                    self.c_grads = tf.gradients(self.c_loss, self.c_params) #c的梯度

            with tf.name_scope('sync'): #同步
                with tf.name_scope('pull'): #把大腦最新的秘笈拉到worker
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'): #把local經歷推送給大腦(並沒有更新local net)
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope): #建立net
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'): #actor net
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'): #critic net
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local 推送經歷上去
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local 把秘笈複製下來
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


######################## Worker #############################
class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped #設定環境
        self.name = name
        self.AC = ACNet(name, globalAC) #local AC：跟global AC有關

    def work(self): #worker的工作：RL更新
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], [] #s,a,r的緩存
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP: #在每個epsoide進行學習
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render() #顯示在電腦上(可以不要)
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a) #done：是否該結束
                done = True if ep_t == MAX_EP_STEP - 1 else False  

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize

                #學習
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done: #該結束
                        v_s_ = 0   # terminal 對未來的期望=0
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0] #用AC net的價值(V)來分析這一步的價值
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_ # gamma discount
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict) #更新到大腦
                    buffer_s, buffer_a, buffer_r = [], [], [] #清空緩存
                    self.AC.pull_global() #提取大腦最新的參數下來

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break


######################## Main #############################
if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"): #大腦 Golbal_Net 設定
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA') #actor更新，使用RMSPropOptimizer
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC') #critic更新，使用RMSPropOptimizer
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # only need its params #創建Golbal Net (ACNet有分global、local)
        
        # Create worker 創建分身(worker)
        workers = []
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC)) #worker class

    COORD = tf.train.Coordinator() #多線程的調度器
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH: #輸出圖
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers: #對於每個worker
        job = lambda: worker.work() #工作
        t = threading.Thread(target=job) #把工作安排到每個線程裡
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads) #join：當每個worker的線程都運行完畢，才往下執行
    #若少了這一行，其中一個worker運行完成了，主程序就會往下走

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

