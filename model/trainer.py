#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/11 16:02
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : trainer.py
# @Software: PyCharm
from data_helper import data_helper
from oqmrc_model import Oqmrc_Model
import tensorflow as tf
from functions import *

if __name__=='__main__':
    dh=data_helper()
    config={'hidden_size':128,
        'num_layers':2,
        'word_dim':128,
        'voc_len':dh.voc_len,
        'max_len':30,
        'lr':0.6,
        'max_grad_norm':5}
    oqmrc_model=Oqmrc_Model(config)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        for i in range(100000):
            data_dic=dh.next_batch(100)

            sess.run(oqmrc_model.train_op,feed_dict={oqmrc_model.query_input:data_dic['query'],
                                                    oqmrc_model.query_len_input:data_dic['query_len'],
                                                    oqmrc_model.passage_input:data_dic['passage'],
                                                    oqmrc_model.passage_len_input:data_dic['passage_len'],
                                                    oqmrc_model.alternatives_input:data_dic['alternative'],
                                                    oqmrc_model.alternatives_len_input: data_dic['alternative_len'],
                                                    oqmrc_model.y_input:data_dic['answer']})
            if i%100==0:
                data_dic=dh.get_val_data()
                loss,score=sess.run([oqmrc_model.loss,oqmrc_model.middle_out],feed_dict={oqmrc_model.query_input:data_dic['query'],
                                                    oqmrc_model.query_len_input:data_dic['query_len'],
                                                    oqmrc_model.passage_input:data_dic['passage'],
                                                    oqmrc_model.passage_len_input:data_dic['passage_len'],
                                                    oqmrc_model.alternatives_input:data_dic['alternative'],
                                                    oqmrc_model.alternatives_len_input:data_dic['alternative_len'],
                                                    oqmrc_model.y_input:data_dic['answer']})
                lookup(score,data_dic['answer'])


