"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__( #初始化
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self): # 建立Policy Gradient NN (2層fc, 1個softmax)
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense( #全連接層(fully-connected)
            inputs=self.tf_obs,
            units=10, #問題不是很複雜，10個神經元就夠了
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense( #輸出所有行為的值 (all_act)
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        #使用softmax
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  #把輸出(action)值轉換成機率，ex: 60%向上、40%向下

        #計算誤差loss
        with tf.name_scope('loss'): 
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action (tf的softmax)
            # or in this way: (莫凡建議使用)
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1) 
            # tf.log(all_act_prob)：選擇每個動作的機率。 #one_hot：把tf_acts變成矩陣形式，才能跟機率矩陣相乘(所選動作為1，其他為0)  #相乘結果為：對應(所選)動作的機率
            # 前面加負號原因：tf只能最小化 minimize(loss)，但我希望獎勵(log_p*V)最大化。
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss=梯度下降趨勢

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss) #tf只有最小化

    def choose_action(self, observation): #直接用輸出得到的機率，來選擇動作
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob (np的func：根據機率選擇矩陣中的數字)
        return action

    def store_transition(self, s, a, r): #儲存回合transition (放進列表)
        self.ep_obs.append(s) #observation 觀測值
        self.ep_as.append(a) #action 所選動作
        self.ep_rs.append(r) #reward 獲得獎勵

    def learn(self): #學習更新參數
        # discount and normalize episode reward (先處理一下reward)
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode (train_op 輸入參數：觀測值、所選動作、獎勵值)
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs] (tf_obs=觀測值)
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ] (tf_acts=所選動作)
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ] (tf_vt=獎勵值)
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data (回合結束，清空列表)
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self): #衰減回合的reward (正規劃)
        # 遊戲1：V漸漸減小，告訴NN後期更新幅度小一點 (要丟下槌子了)
        # 遊戲2：V漸漸增大，告訴NN後期更新幅度大一點 (一開始的移動影響不大，最後快碰到旗子，動大一點)

        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



