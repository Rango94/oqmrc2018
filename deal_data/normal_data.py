#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/10 20:04
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : normal_data.py
# @Software: PyCharm

'''
将需要使用的字段提取出来
'''
import json
import jieba as jb


if __name__ =='__main__':
    with open('word_dic','r',encoding='utf-8') as fo:
        word_dic=json.load(fo)

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
        with open('../../DATA/data/' + file.split('/')[-1], 'w',encoding='utf-8') as out_fo:
            with open(file,'r',encoding='utf-8') as fo:
                for line in fo.readlines():
                    line_dic={}
                    data=json.loads(line)
                    keys = ['passage', 'query']
                    for key in keys:
                        line_dic[key]=' '.join([word.strip() for word in jb.cut(data[key])])

                    alternatives = [word.strip() for word in data['alternatives'].split('|')]
                    try:
                        line_dic['answer']=alternatives.index(data['answer'].strip())
                    except:
                        pass
                    line_dic['alternatives']='\t'.join([' '.join([word.strip() for word in jb.cut(sten.strip())]) for sten in alternatives])
                    out_fo.write(json.dumps(line_dic,ensure_ascii=False)+'\n')












