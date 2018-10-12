#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/12 19:54
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : predicter.py
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
        'lr':3.,
        'max_grad_norm':5}
    oqmrc_model=Oqmrc_Model(config)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, file_pre + 'oqmrc_model')

        data_dic=dh.get_val_data()
        loss,score=sess.run([oqmrc_model.loss,oqmrc_model.middle_out],feed_dict={oqmrc_model.query_input:data_dic['query'],
                                            oqmrc_model.query_len_input:data_dic['query_len'],
                                            oqmrc_model.passage_input:data_dic['passage'],
                                            oqmrc_model.passage_len_input:data_dic['passage_len'],
                                            oqmrc_model.alternatives_input:data_dic['alternative'],
                                            oqmrc_model.alternatives_len_input:data_dic['alternative_len'],
                                            oqmrc_model.y_input:data_dic['answer']})
        lookup(score,data_dic['answer'])
        print('loss_on_val_set:%.6f'%loss,'pre_on_val_set:%0.6f'%cont_pre(score,data_dic['answer']))

        data_dic = dh.get_test_data()
        print(data_dic['alternative'])
        score = sess.run(oqmrc_model.middle_out,feed_dict={oqmrc_model.query_input: data_dic['query'],
                                          oqmrc_model.query_len_input: data_dic['query_len'],
                                          oqmrc_model.passage_input: data_dic['passage'],
                                          oqmrc_model.passage_len_input: data_dic['passage_len'],
                                          oqmrc_model.alternatives_input: data_dic['alternative'],
                                          oqmrc_model.alternatives_len_input: data_dic['alternative_len']})

        with open('predict_on_test','w',encoding='utf-8') as fo:
            for idx in range(len(score)):
                print(str(data_dic['query_id'][idx])+'\t'+data_dic['alternative_raw'][idx][np.argmax(score[idx])])
                fo.write(str(data_dic['query_id'][idx])+'\t'+data_dic['alternative_raw'][idx][np.argmax(score[idx])]+'\n')
