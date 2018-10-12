#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/11 14:55
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : data_helper.py
# @Software: PyCharm
import random as rd
import numpy as np
import json

class data_helper:

    def __init__(self):
        self.train_file_path='../../DATA/data/ai_challenger_oqmrc_trainingset_idx.json'
        self.test_file_path='../../DATA/data/ai_challenger_oqmrc_testa_idx.json'
        self.val_file_path='../../DATA/data/ai_challenger_oqmrc_validationset_idx.json'
        self.train_file_fo=open(self.train_file_path,'r',encoding='utf-8')
        self.test_file_fo=open(self.test_file_path,'r',encoding='utf-8')
        self.val_file_fo=open(self.val_file_path,'r',encoding='utf-8')
        self.max_len=30
        self.max_len_al=5
        self.word_dic=json.load(open('../../DATA/data/word_dic','r',encoding='utf-8'))
        self.voc_len=len(self.word_dic)

    def next_batch(self,size):
        query=[]
        query_len=[]
        passage=[]
        passage_len=[]
        alternative=[]
        alternative_len=[]
        answer=[]

        while size>0:
            line=self.train_file_fo.readline()
            if line=='':
                self.train_file_fo.close()
                self.train_file_fo=open(self.train_file_path,'r',encoding='utf-8')
                line=self.train_file_fo.readline()

            line_json=json.loads(line)

            query_single,query_single_len=self.normal_data([int(i) for i in line_json['query'].split(' ')])
            query.append(query_single)
            query_len.append(query_single_len)

            passage_single,passage_single_len=self.normal_data([int(i) for i in line_json['passage'].split(' ')])
            passage.append(passage_single)
            passage_len.append(passage_single_len)

            alternative_single = [[int(i) for i in sten.split(' ')] for sten in line_json['alternatives'].split('\t')]
            alternative_single_len = []
            for idx, each in enumerate(alternative_single):
                sub_alternative_single, sub_alternative_single_len = self.normal_data(each, self.max_len_al)
                alternative_single[idx] = sub_alternative_single
                alternative_single_len.append(sub_alternative_single_len)

            alternative.append(np.array(alternative_single))
            alternative_len.append(np.array(alternative_single_len))

            answer.append(line_json['answer'])

            size-=1
        return {'query':np.array(query),
                'query_len':np.array(query_len),
                'passage':np.array(passage),
                'passage_len':np.array(passage_len),
                'alternative':np.array(alternative),
                'alternative_len':np.array(alternative_len),
                'answer':np.array(answer)}

    def get_test_data(self):
        query = []
        query_len = []
        passage = []
        passage_len = []
        alternative = []
        alternative_len=[]
        alternative_raw=[]
        query_id=[]
        while True:
            line=self.test_file_fo.readline()

            if line=='':
                self.test_file_fo.close()
                self.test_file_fo=open(self.test_file_path,'r',encoding='utf-8')
                break
            line_json = json.loads(line)
            query_single, query_single_len = self.normal_data([int(i) for i in line_json['query'].split(' ')])
            query.append(query_single)
            query_len.append(query_single_len)

            passage_single, passage_single_len = self.normal_data([int(i) for i in line_json['passage'].split(' ')])
            passage.append(passage_single)
            passage_len.append(passage_single_len)

            alternative_single = [[int(i) for i in sten.split(' ')] for sten in line_json['alternatives'].split('\t')]
            alternative_single_len = []
            for idx, each in enumerate(alternative_single):
                sub_alternative_single, sub_alternative_single_len = self.normal_data(each, self.max_len_al)
                alternative_single[idx] = sub_alternative_single
                alternative_single_len.append(sub_alternative_single_len)

            if np.array(alternative_single).shape!=(3,10):
                print(line_json['alternatives_raw'],line_json['query_id'])
            alternative.append(np.array(alternative_single))
            alternative_len.append(np.array(alternative_single_len))

            alternative_raw.append(line_json['alternatives_raw'])
            query_id.append(line_json['query_id'])

        return {'query': np.array(query),
                'query_len': np.array(query_len),
                'passage': np.array(passage),
                'passage_len': np.array(passage_len),
                'alternative': np.array(alternative),
                'alternative_len':np.array(alternative_len),
                'alternative_raw': alternative_raw,
                'query_id':query_id}

    def get_val_data(self):
        query = []
        query_len = []
        passage = []
        passage_len = []
        alternative = []
        alternative_len=[]
        answer = []

        while True:
            line=self.val_file_fo.readline()
            if line=='':
                self.val_file_fo.close()
                self.val_file_fo=open(self.val_file_path,'r',encoding='utf-8')
                break
            line_json = json.loads(line)
            query_single, query_single_len = self.normal_data([int(i) for i in line_json['query'].split(' ')])
            query.append(query_single)
            query_len.append(query_single_len)

            passage_single, passage_single_len = self.normal_data([int(i) for i in line_json['passage'].split(' ')])
            passage.append(passage_single)
            passage_len.append(passage_single_len)

            alternative_single = [[int(i) for i in sten.split(' ')] for sten in line_json['alternatives'].split('\t')]
            alternative_single_len=[]
            for idx,each in enumerate(alternative_single):
                sub_alternative_single,sub_alternative_single_len=self.normal_data(each,self.max_len_al)
                alternative_single[idx]=sub_alternative_single
                alternative_single_len.append(sub_alternative_single_len)


            alternative.append(np.array(alternative_single))
            alternative_len.append(np.array(alternative_single_len))

            answer.append(line_json['answer'])

        return {'query': np.array(query),
                'query_len': np.array(query_len),
                'passage': np.array(passage),
                'passage_len': np.array(passage_len),
                'alternative': np.array(alternative),
                'alternative_len':np.array(alternative_len),
                'answer': np.array(answer)}

    def normal_data(self,data,max_len=-1):
        if max_len==-1:
            max_len=self.max_len
        data_len=min(len(data),max_len)
        while len(data)<max_len:
            data.append(0)
        while len(data)>max_len:
            data.pop(-1)
        return np.array(data),data_len

if __name__=='__main__':
    dh=data_helper()
    test=dh.get_test_data()
    print(test['alternative'].shape)
    val=dh.get_val_data()
    print(val['alternative'].shape)







