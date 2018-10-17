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

jb.add_word('不')
jb.add_word('无法确定')

if __name__ =='__main__':
    with open('../../DATA/data/word_dic','r',encoding='utf-8') as fo:
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
        with open('../../DATA/data/' + file.split('/')[-1], 'w',encoding='utf-8') as out_fo_0:
            with open('../../DATA/data/'+file.split('/')[-1].replace('.json','_idx.json'),'w',encoding='utf-8') as out_fo_1:
                with open(file,'r',encoding='utf-8') as fo:
                    for line in fo.readlines():
                        line_dic_char={}
                        line_dic_idx={}
                        data=json.loads(line)
                        keys = ['passage', 'query']
                        for key in keys:
                            line_dic_char[key]= ' '.join([word.strip() for word in jb.cut(data[key])])

                            word_list = []
                            for word in line_dic_char[key].split(' '):
                                try:
                                    word_list.append(str(word_dic[word]))
                                except:
                                    print(word)
                                    pass
                            line_dic_idx[key] = ' '.join(word_list)

                        alternatives = [word.strip() for word in data['alternatives'].split('|')]
                        line_dic_char['alternatives_raw'] = alternatives

                        try:
                            line_dic_char['answer']=alternatives.index(data['answer'].strip())
                            line_dic_idx['answer']=line_dic_char['answer']
                        except:
                            pass
                        line_dic_char['alternatives']= '\t'.join([' '.join([word.strip() for word in jb.cut(sten.strip())]) for sten in alternatives])
                        word_list=[]
                        for sten in line_dic_char['alternatives'].strip().split('\t'):
                            word_list.append(' '.join([str(word_dic[word]) for word in sten.split(' ')]))
                        line_dic_idx['alternatives'] = '\t'.join(word_list)
                        line_dic_char['query_id']=data['query_id']

                        line_dic_idx['alternatives_raw']=alternatives
                        line_dic_idx['query_id']=data['query_id']
                        out_fo_0.write(json.dumps(line_dic_char, ensure_ascii=False) + '\n')
                        out_fo_1.write(json.dumps(line_dic_idx, ensure_ascii=False) + '\n')













