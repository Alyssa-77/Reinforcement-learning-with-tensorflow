"""
Reinforcement learning quic example.

state: delay, bandwidth, throughput, loss
reward: delay, throughput, loss

This script is the environment part of this example. The RL is in RL_brain.py.
"""

import numpy as np
import socket
import subprocess, os, time, json

# reward weights
W_LOSS = -0.5    # loss
W_DELAY = -0.5   # delay
W_TPUT = 0    # tput

MAKE_SEC = 16
# LOG_4PERF = "/home/yuki/myquic/msquic/myperf/4perf.txt"

SHELL_RESET = "/home/yuki/myquic/msquic/myshell/server_reset.sh"
SHELL_STREAM = "/home/yuki/myquic/msquic/myshell/server_stream.sh"
# SHELL_CLIENT = "/home/yuki/myquic/msquic/myshell/client.sh"

F_DICT = {0:"0x4000", 1:"0x8000", 2:"0x10000", 3:"0x20000", 4:"0x40000", 5:"0x80000", 6:"0x100000", 7:"0x200000", 8:"0x400000", 9:"0x800000", 10:"0x1000000"}

class Quic(object):
    def __init__(self):
        super(Quic, self).__init__()
        # stream 16KB~16MB (action_space 11 sizes)
        self.action_space = [F_DICT[0], F_DICT[1], F_DICT[2], F_DICT[3], F_DICT[4], F_DICT[5], F_DICT[6], F_DICT[7], F_DICT[8], F_DICT[9], F_DICT[10]] 
        self.n_actions = len(self.action_space) # 11
        self._build_quic()

    def _build_quic(self):
        self.bw = -1
        self.loss = -1
        self.delay = -1
        self.tput = -1
        
    def reset(self): # return 觀測值 (state)
        # self.update()   # 回合結束，更新 todo 有問題 "QUIC has no attribute update???"
        time.sleep(0.5)
        # 回到初始狀態 f=10000
        subprocess.Popen(["sudo", "xterm", "-e", SHELL_RESET]) # open xterm run server_reset.sh (only reset and make) ?
        self.bw = -1
        self.loss = -1
        self.delay = -1
        self.tput = -1
        return self

    def step(self, action): # return s', done, r
        s = [self.bw, self.loss, self.delay, self.tput] # (bodp)
        # 依照action跑quic
        dict_perf = quicServer(action) # ex: action="0x10000"

        # 取得新s，放入s_
        bw = float((dict_perf["bw"].split(" "))[0])          # Mbps
        loss = float((dict_perf["loss"].split(" "))[0])      # %
        delay = float((dict_perf["delay"].split(" "))[0])    # ms
        tput = float((dict_perf["tput"].split(" "))[0])      # kbps
        # print("get perf: bw=",bw,"loss=",loss,"delay=",delay,"tput=",tput)
        s_ = [bw, loss, delay, tput]  # next state (bodp)

        # reward function, done
        if s_[3] == 0:      # todo 何時done? tput=0 不done重做，其他時候done
            reward = 0
            done = True
            s_ = 'terminal'
        else:
            z = zscore(s_[1], s_[2], s_[3]) # s_[bodp]，z=[zo, zd, zp]
            reward = W_LOSS*z[0]+W_DELAY*z[1]+W_TPUT*z[2]   # (-0.5loss, -0.5delay, 0tput) # todo reward設計
            done = False

        return s_, reward, done

    def render(self): # todo 沒動 not sure 
        time.sleep(0.1)
        # self.update()


# todo 沒動 not sure
def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


# run server / client u7 ================================================================
def quicServer(wnd):
    rewrite = rewriteWnd(SHELL_STREAM, wnd)
    if rewrite:
        subprocess.Popen(["sudo", "xterm", "-e", SHELL_STREAM]) # run
        # resetTxt(LOG_4PERF)
        sendData = {"type" : "flag"} # send flag to run client
        all_perf = client_socket('10.0.0.2', 7002, sendData)

        kill_process(SHELL_STREAM.split("/")[-1][0:-3]) # kill? 會不會太早
        rewrite = False
        return all_perf

# socket function u7 ================================================================
def client_socket(host, port, outdata):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    data_string  = json.dumps(outdata) #send data
    s.send(str(data_string).encode())

    indata = s.recv(1024) #get data 
    all_perf = getData(indata)
    s.close()
    return all_perf

def getData(data):
    data = eval(data)
    if data["type"] == "perf":
        perfDict = data["item"][0]    
        return perfDict
        # saveTxt(LOG_4PERF, bw, loss, delay, tput) # save perf -> 4perf.txt

# other function u7 ================================================================
def rewriteWnd(shell, wnd):
    # change SHELL wnd
    string = "define QUIC_DEFAULT_STREAM_FC_WINDOW_SIZE"
    # string = "define QUIC_DEFAULT_CONN_FLOW_CONTROL_WINDOW"
    replacement = ""
    with open(shell, "r") as fp:
        for line in fp:
            if string in line:
                changes = line.replace(line, "sed -i '236a #define QUIC_DEFAULT_STREAM_FC_WINDOW_SIZE      "+wnd+" ' quicdef.h")
                # changes = line.replace(line, "sed -i '246a #define QUIC_DEFAULT_CONN_FLOW_CONTROL_WINDOW   "+wnd+" ' quicdef.h")
                replacement =  replacement + changes + "\n"
            else:
                replacement = replacement+line
    # print(replacement)
    fout = open(shell, "w")
    fout.write(replacement)
    fout.close()
    return True
    
def resetTxt(txt):
    f = open(txt, "w")
    f.write("")
    f.close()

def saveTxt(txt, bw, loss, delay, tput): # 4perf.txt (bw, loss, delay, tput)
    f = open(txt, "w")
    f.write(str(bw)+","+str(loss)+","+str(delay)+","+str(tput)+"\n")
    f.close()

def zscore(o, d, p, wnd): # 固定值下去算，todo 改每輪變化
    avg = [ # avg(o,d,p) (100M, 10^6)
    [0.15, 128.34, 63172.2], [0.66, 141.57, 57546.1], [0.04, 155.57, 59342.0], 
    [0.12, 147.32, 56566.3], [0.49, 123.78, 65289.5], [1.02, 120.21, 66622.1], [0.01, 115.79, 69234.6], 
    [0.59, 117.74, 68099.4], [0.58, 121.33, 66034.0], [0.58, 133.49, 63041.9], [0.63, 127.17, 64200.4]]
    sd = [  # sd(o,d,p) (100M, 10^6)
    [0.35, 16.22, 7278.68], [1.8, 23.53, 6960.61], [0.09, 81.77, 17130.69], 
    [0.31, 37.19, 10091.66], [1.02, 13.91, 6532.64], [3.16, 3.93, 2220.26], [0.04, 5.42, 3422.66], 
    [1.81, 5.72, 3475.82], [1.82, 4.99, 2709.52], [1.82, 41.25, 11041.77], [1.81, 22.07, 8354.89]]
    
    wnd_index = get_key(F_DICT, wnd)
    zd = (d-avg[wnd_index][0])/sd[wnd_index][0]
    zp = (p-avg[wnd_index][1])/sd[wnd_index][1]
    zo = (o-avg[wnd_index][2])/sd[wnd_index][2]
    return [zo, zd, zp]

def kill_process(target):
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    output, error = p.communicate()

    for line in output.splitlines():
        if target in str(line):
            pid = int(line.split(None, 1)[0])
            os.system("sudo kill %s" %(pid))
            
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


if __name__ == '__main__':
    env = Quic()
    # env.after(100, update)  # 時間間隔之後，執行指定的函數
    update()

    # print("QUIC ENV")