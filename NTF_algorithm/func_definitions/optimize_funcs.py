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

# 文字列処理系
import MeCab
import ipadic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#自作
import normalize_funcs
import jntf_dataclasses

eps = float_info.min


#3次元の時間ー単語ー文書テンソルX(時系列方向に重複がない)をNTF
def optimize_doc_joint_NTF(tensors, Kc=15, Kd=15,alpha=10, beta=10, iter_num = 10):

    X = [tensors.X1,tensors.X2]
    T = [X[0].shape[0],X[1].shape[0]]
    I = [X[0].shape[1],X[1].shape[1]]
    J = [X[0].shape[2],X[1].shape[2]]

    # 分解される行列の初期化
    Uc = [cp.random.rand(I[0], Kc), cp.random.rand(I[1], Kc)]
    Ud = [cp.random.rand(I[0], Kd), cp.random.rand(I[1], Kd)]
    Vc = [cp.random.rand(J[0], Kc), cp.random.rand(J[1], Kc)]
    Vd = [cp.random.rand(J[0], Kd), cp.random.rand(J[1], Kd)]
    Wc = [cp.random.rand(T[0], Kc), cp.random.rand(T[1], Kc)]
    Wd = [cp.random.rand(T[0], Kd), cp.random.rand(T[1], Kd)]

    doc_to_time = [tensors.doc_to_time1, tensors.doc_to_time2]
    time_to_doc = [tensors.time_to_doc1, tensors.time_to_doc2]

    memo = []
    loss = calcLoss_doc_joint_NTF(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc,alpha, beta)
    print("loss of start : ", loss)
    # memo.append(loss)
    
    inv_ind = {0:1,1:0}
    for iter in range(iter_num):
        for n in range(2):
            inv = inv_ind[n]

            # Ucの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Uc_num = cp.einsum('tij,jk,tk->ik',X[n],Vc[n],Wc[n])/J[n] + alpha*Uc[inv]
            Uc_den = cp.einsum('tij,jk,tk->ik',doc_product,Vc[n],Wc[n])/J[n] + alpha*Uc[n]
            Uc[n] *=  Uc_num  / (Uc_den + eps)

            # Udの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Ud_num = cp.einsum('tij,jk,tk->ik',X[n],Vd[n],Wd[n]) /J[n]
            Ud_den = cp.einsum('tij,jk,tk->ik',doc_product,Vd[n],Wd[n])/J[n] + beta * cp.tile(cp.sum(Ud[inv],axis=1), (Kd,1)).T
            Ud[n] *=  Ud_num  / (Ud_den + eps)

            # Vcの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Vc_num = cp.einsum('tij,ik,tk->jk',X[n],Uc[n],Wc[n])/J[n]
            Vc_den = cp.einsum('tij,ik,tk->jk',doc_product,Uc[n],Wc[n])/J[n]
            Vc[n] *=  Vc_num  / (Vc_den + eps)
            
            # Vdの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Vd_num = cp.einsum('tij,ik,tk->jk',X[n],Ud[n],Wd[n])/J[n]
            Vd_den = cp.einsum('tij,ik,tk->jk',doc_product,Ud[n],Wd[n])/J[n]
            Vd[n] *=  Vd_num  / (Vd_den + eps)


            # Wcの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Wc_num = cp.einsum('tij,ik,jk->tk',X[n],Uc[n],Vc[n])/J[n]
            Wc_den = cp.einsum('tij,ik,jk->tk',doc_product,Uc[n],Vc[n])/J[n]
            Wc[n] *=  Wc_num  / (Wc_den + eps)

            # Wdの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Wd_num = cp.einsum('tij,ik,jk->tk',X[n],Ud[n],Vd[n])/J[n]
            Wd_den = cp.einsum('tij,ik,jk->tk',doc_product,Ud[n],Vd[n])/J[n]
            Wd[n] *=  Wd_num  / (Wd_den + eps)
            

        if iter%10 ==0:
            # 損失の単調減少性を確認
            loss = calcLoss_doc_joint_NTF(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc, alpha, beta)
            memo.append(loss)
            print(iter, " : ", loss)


    U = [np.hstack((cp.asnumpy(Uc[0]),cp.asnumpy(Ud[0]))), np.hstack((cp.asnumpy(Uc[1]),cp.asnumpy(Ud[1])))]
    V = [np.hstack((cp.asnumpy(Vc[0]),cp.asnumpy(Vd[0]))), np.hstack((cp.asnumpy(Vc[1]),cp.asnumpy(Vd[1])))]
    W = [np.hstack((cp.asnumpy(Wc[0]),cp.asnumpy(Wd[0]))), np.hstack((cp.asnumpy(Wc[1]),cp.asnumpy(Wd[1])))]
    
    # Uの正規化とWで辻褄合わせ(VやWの正規化と辻褄合わせはデータ解釈時に関数を呼び出すことでそれぞれ行う)
    norm_U = [np.linalg.norm(U[0], ord=2, axis=0), np.linalg.norm(U[1], ord=2, axis=0)]
    W[0] = W[0] * norm_U[0]
    U[0] = U[0] / norm_U[0]
    W[1] = W[1] * norm_U[1]
    U[1] = U[1] / norm_U[1]
    normalize_funcs.normDocMatrixForTime(V[0],W[0],time_to_doc[0])
    normalize_funcs.normDocMatrixForTime(V[1],W[1],time_to_doc[1])

    loss = calcLoss_doc_joint_NTF(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc, alpha, beta)
    print("loss of end : ", loss)
    return U, V, W, memo

    #損失関数値を計算
def calcLoss_doc_NTF(X,U,V,W, time_to_doc):
    T = X.shape[0]
    Y=cp.einsum('ik,jk,tk->tij',U,V,W)
    arrayX = cp.empty(0)
    arrayY = cp.empty(0)
    for t in range(T):
            for j in range(time_to_doc[t], time_to_doc[t+1]):
                arrayX = cp.append(arrayX ,X[t,:,j])
                arrayY = cp.append(arrayY ,Y[t,:,j])
    return cp.linalg.norm(arrayX-arrayY)

def calcLoss_doc_joint_NTF(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc, alpha, beta):
    U = [cp.hstack((Uc[0],Ud[0])), cp.hstack((Uc[1],Ud[1]))]
    V = [cp.hstack((Vc[0],Vd[0])), cp.hstack((Vc[1],Vd[1]))]
    W = [cp.hstack((Wc[0],Wd[0])), cp.hstack((Wc[1],Wd[1]))]
    T = X[0].shape[0]
    Y=[cp.einsum('ik,jk,tk->tij',U[0],V[0],W[0]),cp.einsum('ik,jk,tk->tij',U[1],V[1],W[1])]
    arrayX = [cp.empty(0), cp.empty(0)] 
    arrayY = [cp.empty(0), cp.empty(0)] 
    for n in range(2):
        for t in range(T):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    arrayX[n] = cp.append(arrayX[n] ,X[n][t,:,j])
                    arrayY[n] = cp.append(arrayY[n] ,Y[n][t,:,j])
    loss = cp.linalg.norm(arrayX[0]-arrayY[0])+ cp.linalg.norm(arrayX[1]-arrayY[1])  + alpha* cp.linalg.norm(Uc[0]-Uc[1]) + beta* cp.sum(Ud[0].T.dot(Ud[1]))
    return loss



#3次元の時間ー単語ー文書テンソルX(時系列方向に重複がない)をNTF
def optimize_doc_joint_NTF_(X1,doc_to_time1, time_to_doc1,X2,doc_to_time2,time_to_doc2, Kc=15, Kd=15,alpha=10, beta=10, iter_num = 10):

    X = [X1,X2]
    T = [X[0].shape[0],X[1].shape[0]]
    I = [X[0].shape[1],X[1].shape[1]]
    J = [X[0].shape[2],X[1].shape[2]]

    # 分解される行列の初期化
    Uc = [cp.random.rand(I[0], Kc), cp.random.rand(I[1], Kc)]
    Ud = [cp.random.rand(I[0], Kd), cp.random.rand(I[1], Kd)]
    Vc = [cp.random.rand(J[0], Kc), cp.random.rand(J[1], Kc)]
    Vd = [cp.random.rand(J[0], Kd), cp.random.rand(J[1], Kd)]
    Wc = [cp.random.rand(T[0], Kc), cp.random.rand(T[1], Kc)]
    Wd = [cp.random.rand(T[0], Kd), cp.random.rand(T[1], Kd)]

    doc_to_time = [doc_to_time1, doc_to_time2]
    time_to_doc = [time_to_doc1, time_to_doc2]

    memo = []
    loss = calcLoss_doc_joint_NTF(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc,alpha, beta)
    print("loss of start : ", loss)
    # memo.append(loss)
    
    inv_ind = {0:1,1:0}
    for iter in range(iter_num):
        for n in range(2):
            inv = inv_ind[n]

            # Ucの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Uc_num = cp.einsum('tij,jk,tk->ik',X[n],Vc[n],Wc[n])/J[n] + alpha*Uc[inv]
            Uc_den = cp.einsum('tij,jk,tk->ik',doc_product,Vc[n],Wc[n])/J[n] + alpha*Uc[n]
            Uc[n] *=  Uc_num  / (Uc_den + eps)

            # Udの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Ud_num = cp.einsum('tij,jk,tk->ik',X[n],Vd[n],Wd[n]) /J[n]
            Ud_den = cp.einsum('tij,jk,tk->ik',doc_product,Vd[n],Wd[n])/J[n] + beta * cp.tile(cp.sum(Ud[inv],axis=1), (Kd,1)).T
            Ud[n] *=  Ud_num  / (Ud_den + eps)

            # Vcの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Vc_num = cp.einsum('tij,ik,tk->jk',X[n],Uc[n],Wc[n])/J[n]
            Vc_den = cp.einsum('tij,ik,tk->jk',doc_product,Uc[n],Wc[n])/J[n]
            Vc[n] *=  Vc_num  / (Vc_den + eps)
            
            # Vdの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Vd_num = cp.einsum('tij,ik,tk->jk',X[n],Ud[n],Wd[n])/J[n]
            Vd_den = cp.einsum('tij,ik,tk->jk',doc_product,Ud[n],Wd[n])/J[n]
            Vd[n] *=  Vd_num  / (Vd_den + eps)


            # Wcの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Wc_num = cp.einsum('tij,ik,jk->tk',X[n],Uc[n],Vc[n])/J[n]
            Wc_den = cp.einsum('tij,ik,jk->tk',doc_product,Uc[n],Vc[n])/J[n]
            Wc[n] *=  Wc_num  / (Wc_den + eps)

            # Wdの更新
            product=cp.einsum('ik,jk,tk->tij',Uc[n],Vc[n],Wc[n]) + cp.einsum('ik,jk,tk->tij',Ud[n],Vd[n],Wd[n])
            doc_product = cp.zeros((T[n], I[n], J[n]), float)
            for t in range(T[n]):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    doc_product[t, :, j] = product[t, :, j]    
            Wd_num = cp.einsum('tij,ik,jk->tk',X[n],Ud[n],Vd[n])/J[n]
            Wd_den = cp.einsum('tij,ik,jk->tk',doc_product,Ud[n],Vd[n])/J[n]
            Wd[n] *=  Wd_num  / (Wd_den + eps)
            

        if iter%10 ==0:
            # 損失の単調減少性を確認
            loss = calcLoss_doc_joint_NTF(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc, alpha, beta)
            memo.append(loss)
            print(iter, " : ", loss)


    U = [np.hstack((cp.asnumpy(Uc[0]),cp.asnumpy(Ud[0]))), np.hstack((cp.asnumpy(Uc[1]),cp.asnumpy(Ud[1])))]
    V = [np.hstack((cp.asnumpy(Vc[0]),cp.asnumpy(Vd[0]))), np.hstack((cp.asnumpy(Vc[1]),cp.asnumpy(Vd[1])))]
    W = [np.hstack((cp.asnumpy(Wc[0]),cp.asnumpy(Wd[0]))), np.hstack((cp.asnumpy(Wc[1]),cp.asnumpy(Wd[1])))]
    
    # Uの正規化とWで辻褄合わせ(VやWの正規化と辻褄合わせはデータ解釈時に関数を呼び出すことでそれぞれ行う)
    norm_U = [np.linalg.norm(U[0], ord=2, axis=0), np.linalg.norm(U[1], ord=2, axis=0)]
    W[0] = W[0] * norm_U[0]
    U[0] = U[0] / norm_U[0]
    W[1] = W[1] * norm_U[1]
    U[1] = U[1] / norm_U[1]
    normalize_funcs.normDocMatrixForTime(V[0],W[0],time_to_doc[0])
    normalize_funcs.normDocMatrixForTime(V[1],W[1],time_to_doc[1])

    loss = calcLoss_doc_joint_NTF(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc, alpha, beta)
    print("loss of end : ", loss)
    return U, V, W, memo

    #損失関数値を計算
def calcLoss_doc_NTF_(X,U,V,W, time_to_doc):
    T = X.shape[0]
    Y=cp.einsum('ik,jk,tk->tij',U,V,W)
    arrayX = cp.empty(0)
    arrayY = cp.empty(0)
    for t in range(T):
            for j in range(time_to_doc[t], time_to_doc[t+1]):
                arrayX = cp.append(arrayX ,X[t,:,j])
                arrayY = cp.append(arrayY ,Y[t,:,j])
    return cp.linalg.norm(arrayX-arrayY)

def calcLoss_doc_joint_NTF_(X,Uc,Ud,Vc,Vd,Wc,Wd, time_to_doc, alpha, beta):
    U = [cp.hstack((Uc[0],Ud[0])), cp.hstack((Uc[1],Ud[1]))]
    V = [cp.hstack((Vc[0],Vd[0])), cp.hstack((Vc[1],Vd[1]))]
    W = [cp.hstack((Wc[0],Wd[0])), cp.hstack((Wc[1],Wd[1]))]
    T = X[0].shape[0]
    Y=[cp.einsum('ik,jk,tk->tij',U[0],V[0],W[0]),cp.einsum('ik,jk,tk->tij',U[1],V[1],W[1])]
    arrayX = [cp.empty(0), cp.empty(0)] 
    arrayY = [cp.empty(0), cp.empty(0)] 
    for n in range(2):
        for t in range(T):
                for j in range(time_to_doc[n][t], time_to_doc[n][t+1]):
                    arrayX[n] = cp.append(arrayX[n] ,X[n][t,:,j])
                    arrayY[n] = cp.append(arrayY[n] ,Y[n][t,:,j])
    loss = cp.linalg.norm(arrayX[0]-arrayY[0])+ cp.linalg.norm(arrayX[1]-arrayY[1])  + alpha* cp.linalg.norm(Uc[0]-Uc[1]) + beta* cp.sum(Ud[0].T.dot(Ud[1]))
    return loss
