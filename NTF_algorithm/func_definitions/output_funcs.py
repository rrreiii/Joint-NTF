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
import tensorizer

eps = float_info.min

def make_topic_weight_df(U,K,terms,head_num):
    topic_weight_df = pd.DataFrame(index=range(head_num), columns=[])
    for i in range(K):
        topic = pd.DataFrame(terms)
        topic = topic.rename(columns={0:"topic"+str(i)+"の単語"})
        topic["topic"+str(i)+"の重み"] = pd.DataFrame(U[:,i])
        topic = topic.sort_values(by = "topic"+str(i)+"の重み",ascending=False).head(head_num)
        topic.index = range(head_num)
        topic_weight_df = pd.concat([topic_weight_df, topic],axis=1)
    return topic_weight_df


def make_document_weight_df(V,W,X,K,df,time_to_doc):
    normalize_funcs.normTimeMatrixForDoc(V,W,time_to_doc)
    num =  X.shape[2]
    document_weight_df = pd.DataFrame(index=range(num), columns=[])
    for i in range(K):
        dic = {key:val for key,val in zip(range(num),V[:,i])}
        dic2 = dict(sorted(dic.items(), key=lambda x:x[1],reverse=True))
        document_topic = pd.DataFrame(list(map(lambda x:df.iat[x, df.columns.get_loc('number')] ,dic2.keys())))
        document_topic = document_topic.rename(columns={0:"topic"+str(i)+"の文書number"}) #元のエクセルでの番号
        document_topic["topic"+str(i)+"の重み"] = pd.DataFrame(dic2.values())
        document_topic.index = range(num)
        document_weight_df = pd.concat([document_weight_df, document_topic],axis=1)
    return document_weight_df


   #単語集csvを作るためのもの
def make_terms_df(terms,df1,df2,my_tensorizer):
    docs1 = np.array(df1['sentence'])
    docs2 = np.array(df2['sentence'])
    count_vectrizer = CountVectorizer(
        vocabulary = terms,
        tokenizer = my_tensorizer.tokenize_rm_setsubi) #上でつくった関数
    c_matrix = count_vectrizer.transform(np.hstack((docs1,docs2))).toarray()
    c_df = pd.DataFrame(columns=[])
    c_df["terms"] = terms
    c_df["document_frequency"] =  np.count_nonzero(c_matrix.T>=1, axis=1)
    c_df["max_term_frequency"]  = c_matrix.T.max(axis=1)
    c_df["sumof_term_frequency"] = c_matrix.T.sum(axis=1)
    c_df = c_df.sort_values('document_frequency', ascending=False) #ソートでなければ消す
    c_df_allterm_csv = c_df
    print(pd.concat([df1["number"],df2["number"]], axis=0))
    c_df_allterm_df = pd.concat([c_df_allterm_csv, pd.DataFrame(c_matrix.T,columns=pd.concat([df1["number"],df2["number"]], axis=0))], axis=1)
    c_df_allterm_df = c_df_allterm_csv.sort_values('document_frequency', ascending=False) #ソートでなければ消す
    return c_df_allterm_df


def make_comparison_graph(X1,V,W,c,d,time_to_doc1,time_to_doc2):
    K = c+d
    col1 = sns.color_palette("dark", c) +sns.color_palette("muted", d)
    col2 = sns.color_palette("dark", c) +sns.color_palette("colorblind", 100)[5:15]
    T = X1.shape[0]
    t_list = list(range(1,T+1))

    # データ生成
    x = np.linspace(0, 10, 100)
    y1 = x**2
    y2 = x
    y3 = x*2
    y4 = x*2

    T=W[0].shape[0]

    # プロット領域(Figure, Axes)の初期化
    fig = plt.figure(figsize=(50, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.grid()
    ax2.grid()

    # 水平線、垂直線を入れる\
    V[0],W[0] = normalize_funcs.normDocMatrixForTime(V[0],W[0],time_to_doc1)
    V[1],W[1] = normalize_funcs.normDocMatrixForTime(V[1],W[1],time_to_doc2)
    ax1.stackplot(t_list , W[0].T, colors= col1)
    ax2.stackplot(t_list , W[1].T,colors= col2)
    ax1.legend(list(range(0,K)))
    ax2.legend(list(range(0,K)))
    ax1.set_xticks(range(1,T+1))
    ax2.set_xticks(range(1,T+1))
    ax1.set_xlabel("year", fontsize=22)
    ax1.set_ylabel("weight", fontsize=22)
    ax2.set_xlabel("year", fontsize=22)
    ax2.set_ylabel("weight", fontsize=22)
    ax1.set_title("domain1", fontsize=26)
    ax2.set_title("domain2", fontsize=26)
    # plt.show()

    return fig