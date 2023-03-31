# 標準ライブラリ
from sys import float_info
import os
from datetime import datetime as dt
import re
import itertools
import collections

# データ分析系
import numpy as np
import cupy as cp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

eps = float_info.min

def normTimeMatrixForDoc(V,W,time_to_doc):
    T = W.shape[0]
    K = W.shape[1]
    for k in range(K):
        for t in range(T):
            V[time_to_doc[t]: time_to_doc[t+1], k]  *=  W[t, k]
    W //= W
    W[np.isnan(W)]=0
    
    return V,W

def normDocMatrixForTime(V,W,time_to_doc):
    T = W.shape[0]
    K = W.shape[1]
    pr = np.zeros((T, K), float)
    for k in range(K):
        for t in range(T):
            # print("T",T)
            # print("t",t)
            pr[t, k] = V[time_to_doc[t]: time_to_doc[t+1], k].sum()
            for j in range(time_to_doc[t], time_to_doc[t+1]):
                if pr[t, k] != 0:
                    V[j,k] /= pr[t, k]
    W  *=  pr
    return V,W