#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/10 20:03
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : build_dic.py
# @Software: PyCharm
'''
建立字典
'''
import json
import jieba as jb
jb.add_word('不')
jb.add_word('无法确定')
#332270
#331871
#321301
#321308
word_dic={}
n=0
word_dic['##1']=n
n+=1
word_dic['##2']=n
n+=1
word_dic['##3']=n
n+=1
word_dic['##4']=n
n+=1
word_dic['##5']=n
n+=1

filelist=['../../DATA/ai_challenger_oqmrc2018_trainingset_20180816/'
          'ai_challenger_oqmrc_trainingset_20180816/'
          'ai_challenger_oqmrc_trainingset.json',
          '../../DATA/ai_challenger_oqmrc2018_testa_20180816/'
          'ai_challenger_oqmrc_testa_20180816/'
          'ai_challenger_oqmrc_testa.json',
          '../../DATA/ai_challenger_oqmrc2018_validationset_20180816'
          '/ai_challenger_oqmrc_validationset_20180816/'
          'ai_challenger_oqmrc_validationset.json']

for file in filelist:
    with open(file,'r',encoding='utf-8') as train_file:
        for line in train_file.readlines():
            data=json.loads(line)
            keys=['passage','query']
            for key in keys:
                for word in jb.cut(data[key]):
                    word=word.strip()
                    if word not in word_dic:
                        word_dic[word]=n
                        n+=1
                    else:
                        pass
            for sten in data['alternatives'].split('|'):
                for word in jb.cut(sten):
                    word = word.strip()
                    if word not in word_dic:
                        word_dic[word] = n
                        n += 1
                    else:
                        pass

print(len(word_dic))

with open('../../DATA/data/word_dic','w',encoding='utf-8') as fo:
    json.dump(word_dic,fo,ensure_ascii=False)



