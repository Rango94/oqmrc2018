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
    config={'hidden_size':128,
        'num_layers':1,
        'word_dim':128,
        'voc_len':dh.voc_len,
        'max_len_passage':dh_config['max_len_passage'],
        'max_len_query':dh_config['max_len_query'],
        'max_len_alternatives':dh_config['max_len_alternatives'],
        'lr':0.5,
        'max_grad_norm':5,
        'model_name':'THE_lalalast'}

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
    elif config['model_name']=='MANM_6':
        oqmrc_model = MANM_6_Model(config)
    elif config['model_name']=='MANM_7':
        oqmrc_model = MANM_7_Model(config)
    elif config['model_name'] == 'MANM_8':
        oqmrc_model = MANM_8_Model(config)
    elif config['model_name'] == 'CNN':
        oqmrc_model = CNN_model(config)
    elif config['model_name'] == 'ATT':
        oqmrc_model = ATT_model(config)
    elif config['model_name']=='THE_last':
        oqmrc_model=THE_last_attempt(config)
    elif config['model_name']=='THE_lalast':
        oqmrc_model=THE_lalast_attempt(config)
    elif config['model_name']=='THE_lalalast':
        oqmrc_model=THE_lalalast_attempt(config)

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
        grads=0
        for i in range(100000):
            if i%40==0:
                data_dic = dh.get_val_data()
                out=oqmrc_model.val_(sess,[oqmrc_model.loss,oqmrc_model.middle_out,oqmrc_model.lr,oqmrc_model.loss_l2],feed_dict=data_dic, flag=True if ((int)(i/100))%2==0 else False)
                for tmp in oqmrc_model.val_(sess,[oqmrc_model.tmp],feed_dict=data_dic,flag=True if ((int)(i/100))%2==0 else False):
                    print(tmp[0][0][:10])
                    break
                loss=0
                pre=0
                lr=0
                loss_l2=0
                for loss_,score,lr_,loss_l2_ in out:
                    pre_=cont_pre(score,data_dic['answer'])
                    print(pre_)
                    pre+=pre_
                    loss+=loss_
                    lr+=lr_
                    loss_l2+=loss_l2_
                pre/=len(out)
                loss/=len(out)
                loss_l2/=len(out)
                lr/=len(out)

                # loss, score, lr, loss_l2=sess.run([oqmrc_model.loss,oqmrc_model.middle_out,oqmrc_model.lr,oqmrc_model.loss_l2],
                #                                      feed_dict={oqmrc_model.query_input: data_dic['query'],
                #                                                 oqmrc_model.query_len_input: data_dic['query_len'],
                #                                                 oqmrc_model.passage_input: data_dic['passage'],
                #                                                 oqmrc_model.passage_len_input: data_dic['passage_len'],
                #                                                 oqmrc_model.alternatives_input: data_dic['alternative'],
                #                                                 oqmrc_model.alternatives_len_input: data_dic['alternative_len'],
                #                                                 oqmrc_model.y_input: data_dic['answer'],
                #                                                 oqmrc_model.keep_pro: 1,
                #                                                 oqmrc_model.whether_train: -1})
                # pre=cont_pre(score, data_dic['answer'])


                lookup(score, data_dic['answer'],10)
                print('loss_on_training_set:%.6f' % loss_on_train, '|',
                      'loss_on_val_set:%.6f' % loss, '|',
                      'lossl2_on_val_set:%.6f' % loss_l2, '|',
                      'pre_on_val_set:%0.6f' % pre, '|',
                      'lr:%0.6f' % lr, '|',
                      'best:%.6f' % save_mark)
                # try:
                #     print(grads[:3])
                # except:
                #     print(grads)

                if pre>0.77:
                    pre=0
                if pre>save_mark:
                    # att_weight=sess.run(oqmrc_model.attention_weight)
                    # word_embedding=sess.run(oqmrc_model.word_embedding)
                    # np.save('../../DATA/data/att_weight.npy', att_weight)
                    # np.save('../../DATA/data/embedding_trained.npy',word_embedding)
                    saver.save(sess, file_pre+config['model_name']+'_model')
                    save_mark=pre

            data_dic=dh.next_batch(733)
            _,loss_on_train,grads=sess.run([oqmrc_model.train_op,oqmrc_model.loss,oqmrc_model.grads],feed_dict={oqmrc_model.query_input:data_dic['query'],
                                                    oqmrc_model.query_len_input:data_dic['query_len'],
                                                    oqmrc_model.passage_input:data_dic['passage'],
                                                    oqmrc_model.passage_len_input:data_dic['passage_len'],
                                                    oqmrc_model.alternatives_input:data_dic['alternative'],
                                                    oqmrc_model.alternatives_len_input: data_dic['alternative_len'],
                                                    oqmrc_model.y_input:data_dic['answer'],
                                                    oqmrc_model.keep_pro:0.8,
                                                    oqmrc_model.whether_train:-1})

    # writer.close()
