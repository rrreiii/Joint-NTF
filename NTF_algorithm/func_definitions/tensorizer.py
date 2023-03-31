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
import jntf_dataclasses

eps = float_info.min


class DocumentJntfTensorizer:
    def __init__(self,df1,df2,exceptype,excepword,max_df, min_df,time_span):
        self.df1 = df1
        self.df2 = df2
        self.exceptype = exceptype
        self.excepword = excepword
        self.max_df = max_df
        self.min_df = min_df
        self.time_span = time_span

    # mecabの形態素解析により,一般的すぎない連名詞の辞書を作成。make_doc_matrices関数内で用いる
    def tokenize_rm_setsubi(self,sentence):
        """ 連名詞のリストをトークンとして返す """
        sentence  = re.sub('\d+', '', sentence)   # 数字を削除(macabでなぜか名詞に認定された数字あったからここで消しておく)
        sentence  = text=re.sub(r'[ -/:-@\[-~]', "。", sentence)#半角記号,数字,英字
        sentence  = text=re.sub(r'[！-／：-＠［-｀｛-～、-〜”’%]+/g', "。", sentence)#全角記号
        wakati = MeCab.Tagger(ipadic.MECAB_ARGS)
        node = wakati.parseToNode(sentence)
        sequence = 2   #何連名詞以上をかえすか
        dictionary = []
        prev_seq = 0
        flag = 0
        while node:
            word = node.surface
            hinshi = node.feature.split(",")

            if prev_seq  : # その前に名詞が1以上つづいていたら
                if  (hinshi[0] == "名詞" ) and  (hinshi[0] not in self.exceptype) and  (hinshi[1] not in self.exceptype )and (hinshi[2] not in self.exceptype )and (word not in self.excepword) : #今回、除外名詞以外だったら、
                    dictionary[-1] = dictionary[-1]+word   # 辞書の最後の単語にこの単語を結合
                    prev_seq  +=1
                    flag = 0
                    
                elif (hinshi[1] == "接尾" ): #接尾辞だったら
                    dictionary[-1] = dictionary[-1]+word   # とりあえず辞書の最後の単語にこの単語を結合して
                    prev_seq  +=1
                    flag = 1 #フラグを1に

                elif flag == 1: # 一個前が接尾辞で今度が名詞でなかったら、一個前の接尾辞で終わってる単語を辞書から削除
                    dictionary.pop(-1)   #一個前の接尾辞で終わってる単語を辞書から削除
                    prev_seq  =0
                    flag = 0
                else:
                    flag = 0
                    if prev_seq  < sequence: 
                        dictionary.pop(-1) #一個前の単語を辞書から削除
                    prev_seq = 0
            else:  # その前に名詞がつづいてなくて
                if  (hinshi[0] == "名詞" or hinshi[0] == "接頭辞")  and  hinshi[0] not in self.exceptype and  hinshi[1] not in self.exceptype and hinshi[2] not in self.exceptype and word not in self.excepword :
                    dictionary.append(word)
                    prev_seq += 1
                    flag = 0
            node = node.next 
        return dictionary


    # 単語ー文書行列の作成
    def make_doc_matrices(self):
        docs1 = np.array(self.df1['sentence'])
        docs2 = np.array(self.df2['sentence'])
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer = self.tokenize_rm_setsubi,
            lowercase=True,
            max_df=self.max_df,
            min_df=self.min_df)
        tfidf_vectorizer.fit(np.hstack((docs1,docs2)))
        terms= tfidf_vectorizer.get_feature_names()
        matrix1 = tfidf_vectorizer.transform(docs1).toarray().T
        matrix2 = tfidf_vectorizer.transform(docs2).toarray().T
        

        return terms,matrix1,matrix2


    # 時間ー単語ー文書テンソルに変形
    def make_doc_tensors_from_matrices(self,matrix1,matrix2):
        time_doc1 = [0]+list(pd.concat([self.df1,self.df2]).resample(self.time_span).count()["id_a"])
        time_to_doc1 = list(itertools.accumulate(time_doc1))
        doc_to_time1 = []
        for i in range(0,len(time_to_doc1)):
            doc_to_time1+=[i-1]*time_doc1[i]
        T = len(time_to_doc1)-1
        I  = matrix1.shape[0]
        J = matrix1.shape[1]
        X1 = np.zeros((T, I, J), float)
        for t in range(T):
                for j in range(time_to_doc1[t], time_to_doc1[t+1]):
                    X1[t,:,j] = matrix1[:,j]

        # 時間ー単語ー文書テンソルに変形
        time_doc2 = [0]+list(pd.concat([self.df1,self.df2]).resample(self.time_span).count()["id_b"])
        time_to_doc2 = list(itertools.accumulate(time_doc2))
        doc_to_time2 = []
        for i in range(0,len(time_to_doc2)):
            doc_to_time2+=[i-1]*time_doc2[i]
        T = len(time_to_doc2)-1
        I  = matrix2.shape[0]
        J = matrix2.shape[1]
        X2 = np.zeros((T, I, J), float)
        for t in range(T):
                for j in range(time_to_doc2[t], time_to_doc2[t+1]):
                    X2[t,:,j] = matrix2[:,j]

        return time_to_doc1,time_to_doc2,doc_to_time1,doc_to_time2,X1,X2

    # 時間ー単語ー文書テンソルに変形
    def tensorize(self):
        terms,matrix1,matrix2 = self.make_doc_matrices()
        time_to_doc1,time_to_doc2,doc_to_time1,doc_to_time2,X1,X2 = self.make_doc_tensors_from_matrices(matrix1,matrix2)

        tensors = jntf_dataclasses.JntfDocumentTensors(terms,X1,time_to_doc1,doc_to_time1,X2,time_to_doc2,doc_to_time2)

        return tensors

        # return terms,time_to_doc1,time_to_doc2,doc_to_time1,doc_to_time2,X1,X2





