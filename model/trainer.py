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
    file_pre='../../tf_model_file/'
    dh=data_helper()
    config={'hidden_size':128,
        'num_layers':2,
        'word_dim':128,
        'voc_len':dh.voc_len,
        'max_len':30,
        'lr':2.,
        'max_grad_norm':5}
    print('字典长度%d'%dh.voc_len)
    oqmrc_model=Oqmrc_Model(config)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        min_loss=1
        for i in range(100000):
            data_dic=dh.next_batch(500)

            _,loss_on_train=sess.run([oqmrc_model.train_op,oqmrc_model.loss],feed_dict={oqmrc_model.query_input:data_dic['query'],
                                                    oqmrc_model.query_len_input:data_dic['query_len'],
                                                    oqmrc_model.passage_input:data_dic['passage'],
                                                    oqmrc_model.passage_len_input:data_dic['passage_len'],
                                                    oqmrc_model.alternatives_input:data_dic['alternative'],
                                                    oqmrc_model.alternatives_len_input: data_dic['alternative_len'],
                                                    oqmrc_model.y_input:data_dic['answer']})
            if i%100==0:
                data_dic=dh.get_val_data()
                loss,score,lr=sess.run([oqmrc_model.loss,oqmrc_model.middle_out,oqmrc_model.lr],feed_dict={oqmrc_model.query_input:data_dic['query'],
                                                    oqmrc_model.query_len_input:data_dic['query_len'],
                                                    oqmrc_model.passage_input:data_dic['passage'],
                                                    oqmrc_model.passage_len_input:data_dic['passage_len'],
                                                    oqmrc_model.alternatives_input:data_dic['alternative'],
                                                    oqmrc_model.alternatives_len_input:data_dic['alternative_len'],
                                                    oqmrc_model.y_input:data_dic['answer']})

                lookup(score,data_dic['answer'])
                print('loss_on_training_set:%.6f'%loss_on_train,'loss_on_val_set:%.6f'%loss,'pre_on_val_set:%0.6f'%cont_pre(score,data_dic['answer']),'lr:%0.6f'%lr)
                if loss<min_loss:
                    saver.save(sess, file_pre+'oqmrc_model')
                    min_loss=loss

