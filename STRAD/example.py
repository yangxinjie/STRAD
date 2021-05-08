from Kitsune import Kitsune
import numpy as np
import time
import pandas as pd

##############################################################################
# Kitsune a lightweight online network intrusion detection system based on an ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn, and detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 3.6.3   #######################

# Load Mirai pcap (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...


# File location
path = "../data/mawilab.pcap" #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)
path = "../data/newlabel3.csv"
X = pd.read_csv(path)
print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    if(X.loc[i,'label']==1):
        continue
    i+=1
    if i==240001:
        break
    rmse = K.proc_next_packet()
    if rmse == -1:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete train. Time elapsed: "+ str(stop - start))
a=np.array(RMSEs)
np.save('trainrmse.npy',a)   # 保存为.npy格式
# 读取
#a=np.load('a.npy')
#a=a.tolist()

start = time.time()
RMSEtest=[]
while(i<300001):
    rmse=K.proc_next_packet()
    i+=1
    RMSEtest.append(rmse)
stop = time.time()
print("Complete test. Time elapsed: "+ str(stop - start))
a=np.array(RMSEtest)
np.save('testrmse.npy',a)   # 保存为.npy格式    
    

# # Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
# from scipy.stats import norm
# benignSample = np.log(RMSEs[FMgrace+ADgrace+1:100000])
# logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# # plot the RMSE anomaly scores
# print("Plotting results")
# from matplotlib import pyplot as plt
# from matplotlib import cm
# plt.figure(figsize=(10,5))
# fig = plt.scatter(range(FMgrace+ADgrace+1,len(RMSEs)),RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
# plt.yscale("log")
# plt.title("Anomaly Scores from Kitsune's Execution Phase")
# plt.ylabel("RMSE (log scaled)")
# plt.xlabel("Time elapsed [min]")
# plt.annotate('Mirai C&C channel opened [Telnet]', xy=(121662,RMSEs[121662]), xytext=(151662,1),arrowprops=dict(facecolor='black', shrink=0.05),)
# plt.annotate('Mirai Bot Activated\nMirai scans network\nfor vulnerable devices', xy=(122662,10), xytext=(122662,150),arrowprops=dict(facecolor='black', shrink=0.05),)
# plt.annotate('Mirai Bot launches DoS attack', xy=(370000,100), xytext=(390000,1000),arrowprops=dict(facecolor='black', shrink=0.05),)
# figbar=plt.colorbar()
# figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
# plt.show()
