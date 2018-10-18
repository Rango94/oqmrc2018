#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/11 16:02
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : trainer.py
# @Software: PyCharm
from data_helper import data_helper
from oqmrc_model import *
import tensorflow as tf
from functions import *
import sys

if __name__=='__main__':
    file_pre='../../tf_model_file/'

    dh_config={'max_len_passage':80,
        'max_len_query':20,
        'max_len_alternatives':5}
    dh = data_helper(dh_config)
    config={'hidden_size':50,
        'num_layers':1,
        'word_dim':128,
        'voc_len':dh.voc_len,
        'max_len_passage':dh_config['max_len_passage'],
        'max_len_query':dh_config['max_len_query'],
        'max_len_alternatives':dh_config['max_len_alternatives'],
        'lr':2.,
        'max_grad_norm':5,
        'model_name':'MANM_4'}

    print('字典长度%d'%dh.voc_len)

    if config['model_name']=='MANM':
        oqmrc_model=MANM_Model(config)
    elif config['model_name']=='WNZ':
        oqmrc_model=WNZ_Model(config)
    elif config['model_name']=='MANM_2':
        oqmrc_model=MANM_2_Model(config)
    elif config['model_name']=='MANM_3':
        oqmrc_model=MANM_3_Model(config)
    elif config['model_name']=='MANM_4':
        oqmrc_model=MANM_4_Model(config)
    elif config['model_name']=='MANM_5':
        oqmrc_model = MANM_5_Model(config)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:

        tf.global_variables_initializer().run()

        # writer = tf.summary.FileWriter('../../tensorboard/tensorboard', sess.graph)
        saver = tf.train.Saver()

        if len(sys.argv)==2 and sys.argv[1]=='clear':
            pass
        else:
            print('loading pretrained model...')
            saver.restore(sess, file_pre + config['model_name']+'_model')
            print('done')

        save_mark=0
        loss_on_train=1
        for i in range(100000):
            if i%100==0:
                data_dic = dh.get_val_data()
                out=oqmrc_model.val_(sess,[oqmrc_model.loss,oqmrc_model.middle_out,oqmrc_model.lr,oqmrc_model.loss_l2],feed_dict=data_dic)
                for tmp in oqmrc_model.val_(sess,[oqmrc_model.tmp],feed_dict=data_dic):
                    print(tmp[0])
                    break
                loss=0
                pre=0
                lr=0
                loss_l2=0
                for loss_,score,lr_,loss_l2_ in out:
                    pre+=cont_pre(score,data_dic['answer'])
                    loss+=loss_
                    lr+=lr_
                    loss_l2+=loss_l2_
                pre/=len(out)
                loss/=len(out)
                loss_l2/=len(out)
                lr/=len(out)
                lookup(score, data_dic['answer'])
                print('loss_on_training_set:%.6f' % loss_on_train, '|',
                      'loss_on_val_set:%.6f' % loss, '|',
                      'lossl2_on_val_set:%.6f' % loss_l2, '|',
                      'pre_on_val_set:%0.6f' % pre, '|',
                      'lr:%0.6f' % lr, '|',
                      'best:%.6f' % save_mark)

                if pre>0.77:
                    pre=0
                if pre>save_mark:
                    saver.save(sess, file_pre+config['model_name']+'_model')
                    save_mark=pre

            data_dic=dh.next_batch(437)
            _,loss_on_train=sess.run([oqmrc_model.train_op,oqmrc_model.loss],feed_dict={oqmrc_model.query_input:data_dic['query'],
                                                    oqmrc_model.query_len_input:data_dic['query_len'],
                                                    oqmrc_model.passage_input:data_dic['passage'],
                                                    oqmrc_model.passage_len_input:data_dic['passage_len'],
                                                    oqmrc_model.alternatives_input:data_dic['alternative'],
                                                    oqmrc_model.alternatives_len_input: data_dic['alternative_len'],
                                                    oqmrc_model.y_input:data_dic['answer'],
                                                    oqmrc_model.keep_pro:0.6,})

    # writer.close()
