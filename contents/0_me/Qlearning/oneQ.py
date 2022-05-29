"""
One file Q learing, base on zscore.py
Run with client.py

After set miniet(bw delay), 輸入f的index, 自動跑10次, 取得dbpo, 存在/myperf/zscore。
f1 = [4000, 8000, 10000, 20000, 40000, 80000, 100000, 200000, 400000, 800000, 1000000] 16KB~16MB
c1 = 16MB = 1000000

改LOG、FLOW_SIZE、SHELL、rewriteWnd(string、changes)
"""

import subprocess, os
import json
import socket
import time
import math

# kill process ================================================================
def kill_process(target):
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    output, error = p.communicate()

    for line in output.splitlines():
        if target in str(line):
            pid = int(line.split(None, 1)[0])
            # os.kill(pid, 9)
            # print("kill process:", target, pid)
            os.system("sudo kill %s" %(pid))
            
# socket function ================================================================
def client_socket(host, port, outdata):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    data_string  = json.dumps(outdata) #send data
    # print('send: ' + data_string)
    s.send(str(data_string).encode())

    indata = s.recv(1024) #get data 
    getData(indata)
    s.close()

def getData(data):
    data = eval(data)
    if data["type"] == "perf":      # get perf
        s = calState(data["item"][0])
        print("get perf: bw=",s['b'],"loss=",s['o'],"delay=",s['d'],"tput=",s['p'])
        if FIRST==False:
            r = calReward(s)

        # (0) save perf as state txt, for real zscore & check performance
        state_txt = LOG+"state.txt"
        appendTxt(state_txt, s['b'], s['o'], s['d'], s['p'])

        # (1) open another h1, close SHELL
        kill_process(STREAM[1])    # kill

        # (3) RL cal a
        a = calAction(s, r)
        print("choose wnd =",a)

        # (4) change SHELL wnd size
        rewriteWnd(STREAM[0], a)   # shell

# RL function ================================================================  
def calState(perfDict): #cal from link status
    # print("state")   
    bw = float((perfDict["bw"].split(" "))[0])          # Mbps
    loss = float((perfDict["loss"].split(" "))[0])      # %
    delay = float((perfDict["delay"].split(" "))[0])    # ms
    tput = float((perfDict["tput"].split(" "))[0])      # kbps
    state = {'b':bw, 'o':loss, 'd':delay, 'p':tput}     #todo 一開始沒收到職怎辦
    return state

def calAction(s):
    # print(wnd)
    wnd="0x20000" # todo
    return wnd

def calReward(s, a):
    # print("reward")
    z = zscore_fix(s['o'], s['d'], s['p'], wnd=a) # z=[zo, zd, zp]
    r = W_LOSS*z[0]+W_DELAY*z[1]+W_TPUT*z[2]   # (-0.5loss, -0.5delay, 0tput) # todo reward設計
    return r

# other function ================================================================
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

def appendTxt(txt, bw, loss, delay, tput): # state
    f = open(txt, "a")
    f.write(str(bw)+","+str(loss)+","+str(delay)+","+str(tput)+"\n")
    f.close()

def resetTxt(txt):
    f = open(txt, "w")
    f.write("bw (Mbps),loss (%),delay (ms),tput (kbps)\n")
    f.close()

def zscore_fix(o, d, p, wnd): # 固定值下去算，todo 改每輪變化
    avg = [ # avg(o,d,p) (100M, 10^6)
    [0.15, 128.34, 63172.2], [0.66, 141.57, 57546.1], [0.04, 155.57, 59342.0], 
    [0.12, 147.32, 56566.3], [0.49, 123.78, 65289.5], [1.02, 120.21, 66622.1], [0.01, 115.79, 69234.6], 
    [0.59, 117.74, 68099.4], [0.58, 121.33, 66034.0], [0.58, 133.49, 63041.9], [0.63, 127.17, 64200.4]]
    sd = [  # sd(o,d,p) (100M, 10^6)
    [0.35, 16.22, 7278.68], [1.8, 23.53, 6960.61], [0.09, 81.77, 17130.69], 
    [0.31, 37.19, 10091.66], [1.02, 13.91, 6532.64], [3.16, 3.93, 2220.26], [0.04, 5.42, 3422.66], 
    [1.81, 5.72, 3475.82], [1.82, 4.99, 2709.52], [1.82, 41.25, 11041.77], [1.81, 22.07, 8354.89]]
    
    wnd_index = get_key(F_DICT, wnd)
    zd = (d-avg[wnd_index][0])/sd[wnd_index][0] # Z=(x-avg)/sd
    zp = (p-avg[wnd_index][1])/sd[wnd_index][1]
    zo = (o-avg[wnd_index][2])/sd[wnd_index][2]
    return [zo, zd, zp]

def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"

# Main function =============================================================
EP = 2         # RL run ep times
MAKE_SEC = 16   # C make second

LOG = "/home/yuki/myquic/msquic/myperf/oneQ/"
F_DICT = {0:"0x4000", 1:"0x8000", 2:"0x10000", 3:"0x20000", 4:"0x40000", 5:"0x80000", 6:"0x100000", 7:"0x200000", 8:"0x400000", 9:"0x800000", 10:"0x1000000"}

STREAM = ["/home/yuki/myquic/msquic/myshell/server_stream.sh", "server_stream"] # shell, kill

# reward weights
W_LOSS = -0.5    # loss
W_DELAY = -0.5   # delay
W_TPUT = 0    # tput


FIRST=True
for i in range(EP): # todo while true or not done
    if i==0:
        print("reset stream window = 0x10000")
        rewriteWnd(STREAM[0], "0x10000")
    else:
        FIRST=False

    subprocess.Popen(["sudo", "xterm", "-e", STREAM[0]]) 
    print("\nwait for make:", MAKE_SEC,"s")
    for j in range(MAKE_SEC):
        time.sleep(1)
    sendData = {"type" : "flag"} # send flag to run client
    client_socket('10.0.0.2', 7002, sendData)