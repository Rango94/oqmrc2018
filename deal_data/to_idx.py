#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/10 21:25
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : to_idx.py
# @Software: PyCharm
'''
将那什么转为索引
'''
import json
if __name__=='__main__':
    filelist=['../../DATA/data/ai_challenger_oqmrc_trainingset.json',
                  '../../DATA/data/ai_challenger_oqmrc_testa.json',
                  '../../DATA/data/ai_challenger_oqmrc_validationset.json']
    with open('../../DATA/data/word_dic','r',encoding='utf-8') as fo:
        word_dic=json.load(fo)

    for file in filelist:
        with open(file.replace('.json','_idx.json'),'w',encoding='utf-8') as out_fo:
            with open(file,'r',encoding='utf-8') as fo:
                for line in fo.readlines():
                    line_dic = {}
                    data = json.loads(line)
                    keys = ['passage', 'query']
                    for key in keys:
                        word_list=[]
                        for word in data[key].strip().split(' '):
                            try:
                                word_list.append(str(word_dic[word]))
                            except:
                                print(word)
                                pass
                        line_dic[key] = ' '.join(word_list)
                    try:
                        line_dic['answer'] = data['answer']
                    except:
                        pass
                    word_list = []
                    for sten in data['alternatives'].strip().split('\t'):
                        word_list.append(' '.join([str(word_dic[word]) for word in sten.split(' ')]))
                    line_dic['alternatives'] = '\t'.join(word_list)
                    line_dic['alternatives_raw']=data['alternatives_raw']
                    line_dic['query_id']=data['query_id']
                    out_fo.write(json.dumps(line_dic,ensure_ascii=False) + '\n')