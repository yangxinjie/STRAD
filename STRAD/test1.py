from Kitsune import Kitsune
import numpy as np
import time
import pandas as pd
import sys
import pandas as pd
import numpy as np
#import pyhash 
import gensim
import multiprocessing as mp
from joblib import Parallel, delayed
import concurrent.futures
from pprint import pprint
import random
import KitNET as kit
import numpy as np
import pandas as pd
import time
#import mpld3 as mp
import re
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import manifold
from sklearn.decomposition import PCA, TruncatedSVD
import csv
import time
import joblib
from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.metrics import confusion_matrix
from sklearn import metrics

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
path = "../data/maliwnormal.csv" #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 40000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 200000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

print("Running Kitsune:")
Xvectors = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i+=1
    if i % 10000 == 0:
        print(i)
    if i == 240001:
        break
    rmse = K.proc_next_packet()
    Xvectors.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))
numpy_array = np.array(Xvectors)
np.save('RMSES240000.npy',numpy_array )
#numpy_array = np.load('log.npy')