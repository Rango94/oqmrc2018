#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/12 19:54
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : predicter.py
# @Software: PyCharm

from data_helper import data_helper
from oqmrc_model import *
import tensorflow as tf
from functions import *

if __name__=='__main__':
    file_pre = '../../tf_model_file/'

    dh_config = {'max_len_passage': 80,
                 'max_len_query': 20,
                 'max_len_alternatives': 5}
    dh = data_helper(dh_config)
    config = {'hidden_size': 128,
              'num_layers': 1,
              'word_dim': 128,
              'voc_len': dh.voc_len,
              'max_len_passage': dh_config['max_len_passage'],
              'max_len_query': dh_config['max_len_query'],
              'max_len_alternatives': dh_config['max_len_alternatives'],
              'lr': 0.5,
              'max_grad_norm': 5,
              'model_name': 'THE_lalalast'}

    print('字典长度%d' % dh.voc_len)

    if config['model_name'] == 'MANM':
        oqmrc_model = MANM_Model(config)
    elif config['model_name'] == 'WNZ':
        oqmrc_model = WNZ_Model(config)
    elif config['model_name'] == 'MANM_2':
        oqmrc_model = MANM_2_Model(config)
    elif config['model_name'] == 'MANM_3':
        oqmrc_model = MANM_3_Model(config)
    elif config['model_name'] == 'MANM_4':
        oqmrc_model = MANM_4_Model(config)
    elif config['model_name'] == 'MANM_5':
        oqmrc_model = MANM_5_Model(config)
    elif config['model_name'] == 'MANM_6':
        oqmrc_model = MANM_6_Model(config)
    elif config['model_name'] == 'MANM_7':
        oqmrc_model = MANM_7_Model(config)
    elif config['model_name'] == 'MANM_8':
        oqmrc_model = MANM_8_Model(config)
    elif config['model_name'] == 'CNN':
        oqmrc_model = CNN_model(config)
    elif config['model_name'] == 'ATT':
        oqmrc_model = ATT_model(config)
    elif config['model_name'] == 'THE_last':
        oqmrc_model = THE_last_attempt(config)
    elif config['model_name'] == 'THE_lalast':
        oqmrc_model = THE_lalast_attempt(config)
    elif config['model_name'] == 'THE_lalalast':
        oqmrc_model = THE_lalalast_attempt(config)


    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        saver.restore(sess, file_pre + config['model_name']+'_model')

        data_dic=dh.get_val_data()
        loss,score=sess.run([oqmrc_model.loss,oqmrc_model.middle_out],feed_dict={oqmrc_model.query_input:data_dic['query'],
                                            oqmrc_model.query_len_input:data_dic['query_len'],
                                            oqmrc_model.passage_input:data_dic['passage'],
                                            oqmrc_model.passage_len_input:data_dic['passage_len'],
                                            oqmrc_model.alternatives_input:data_dic['alternative'],
                                            oqmrc_model.alternatives_len_input:data_dic['alternative_len'],
                                            oqmrc_model.y_input:data_dic['answer'],
                                            oqmrc_model.keep_pro:1})
        lookup(score,data_dic['answer'])
        print('loss_on_val_set:%.6f'%loss,'pre_on_val_set:%0.6f'%cont_pre(score,data_dic['answer']))

        data_dic = dh.get_test_data()
        score = sess.run(oqmrc_model.middle_out,feed_dict={oqmrc_model.query_input: data_dic['query'],
                                            oqmrc_model.query_len_input: data_dic['query_len'],
                                            oqmrc_model.passage_input: data_dic['passage'],
                                            oqmrc_model.passage_len_input: data_dic['passage_len'],
                                            oqmrc_model.alternatives_input: data_dic['alternative'],
                                            oqmrc_model.alternatives_len_input: data_dic['alternative_len'],
                                            oqmrc_model.keep_pro:1})

        with open('predict_on_test','w',encoding='utf-8') as fo:
            for idx in range(len(score)):
                #print(str(data_dic['query_id'][idx])+'\t'+data_dic['alternative_raw'][idx][np.argmax(score[idx])])
                fo.write(str(data_dic['query_id'][idx])+'\t'+data_dic['alternative_raw'][idx][np.argmax(score[idx])]+'\n')
