#!/usr/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/10 21:49
# @Author  : Nanzhi.Wang
# @User    : wnz
# @Site    : https://github.com/rango94
# @File    : oqmrc_model.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
import numpy as np
import random as rd
import math

class WNZ_Model:

    def __init__(self,config):
        self.hidden_size=config['hidden_size']
        self.num_layers=config['num_layers']
        self.word_dim=config['word_dim']
        self.voc_len=config['voc_len']
        self.max_len=config['max_len']
        self.feature_extract_layer_num=3
        self.output_layer_num=6
        self.global_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 800, 0.96, staircase=True)

        self.max_grad_norm=config['max_grad_norm']
        self.auto_NN={}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.forward_step1()
        self.forward_step2()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro=tf.placeholder(dtype=tf.float32,name='keep_pro')

        self.query_input=tf.placeholder(dtype=tf.int32,shape=[None,None],name='query_input')
        self.passage_input=tf.placeholder(dtype=tf.int32,shape=[None,None],name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input=tf.placeholder(dtype=tf.int32,shape=[None],name='y_input')
        self.alternatives_input=tf.placeholder(dtype=tf.int32,shape=[None,3,None],name='alternatives_input')
        self.alternatives_len_input=tf.placeholder(dtype=tf.int32,shape=[None,3],name='alternatives_len_input')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.2, 0.2)
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding=tf.get_variable(dtype=tf.float32,
                                                initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                        dtype=tf.float32),name='word_embedding',trainable=True)
            print('loaded')
        with tf.variable_scope('LSTM_encoder',reuse=None):

            self.query_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro) for _ in
                range(self.num_layers)])

            self.passage_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro) for _ in
                range(self.num_layers)])

            self.alternatives_cell=tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size) for _ in range(1)])

        with tf.variable_scope('attention'):
            self.attention_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size,self.hidden_size],initializer=self.initializer,name='attention_weight')
        with tf.variable_scope('feature_extract'):
            self.generate_NN(self.hidden_size*2,self.hidden_size,self.feature_extract_layer_num,'feature_extract')
        with tf.variable_scope('similar'):
            self.similar_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size,self.hidden_size],name='similar_weight')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*2,1, self.output_layer_num,'output_layer')

    def generate_NN(self,input_size,output_size,layers,name):
        auto_NN_weights=[]
        auto_NN_biases=[]
        for i in range(layers):
            if i==layers-1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32,shape=[input_size,output_size],
                                                       name='auto_NN_weight_'+name+'_'+str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32,shape=[output_size],
                                                      name='auto_NN_bias_'+name+'_'+str(i),initializer=self.initializer))
                # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0000001)(auto_NN_weights[i]))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size , input_size ],
                                                       name='auto_NN_weight_' +name+'_'+ str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size ],
                                                      name='auto_NN_bias_' + name+'_'+str(i),initializer=self.initializer))
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.005)(auto_NN_weights[i]))
        self.auto_NN[name]=zip(auto_NN_weights,auto_NN_biases)

    def forward_step0(self):
        #这一部分主要是lstm编码过程
        query_emb=tf.nn.embedding_lookup(self.word_embedding,self.query_input)
        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)
        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:,0,:])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])
        with tf.variable_scope('query_encoder'):
            self.query_output, self.query_state = dynamic_rnn(self.query_cell, query_emb,sequence_length=self.query_len_input,dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            self.passage_output,self.passage_state=dynamic_rnn(self.passage_cell, passage_emb,sequence_length=self.passage_len_input,dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, self.alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb_0,sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, self.alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb_1,sequence_length=self.alternatives_len_input[:,1],dtype=tf.float32)
            alternatives_output_2, self.alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb_2,sequence_length=self.alternatives_len_input[:,2],dtype=tf.float32)

    def forward_step1(self):
        #这一部分计算attention后并且提取特征后的结果
        self.query_state = self.get_h(self.query_state)
        self.passage_state=self.compute_attention(self.query_state, self.passage_output, self.passage_len_input)

        self.query_passage_concated=tf.concat([self.query_state, self.passage_state], axis=1)
        for idx, (weight, bias) in enumerate(self.auto_NN['feature_extract']):
            if idx%2==0:
                self.query_passage_concated = tf.nn.dropout(tf.nn.elu(tf.matmul(self.query_passage_concated,weight)+bias),
                                                            keep_prob=self.keep_pro)
            else:
                self.query_passage_concated = tf.nn.elu(tf.matmul(self.query_passage_concated, weight) + bias)

    def forward_step2(self):
        #这一部分收尾工作，计算三个答案的具体的值
        # alternatives_state = tf.concat([self.get_h(self.alternatives_state_0),
        #                                 self.get_h(self.alternatives_state_1), self.get_h(self.alternatives_state_2)],
        #                                axis=1)
        # tmp=tf.matmul(tf.reshape(alternatives_state,shape=[-1,self.hidden_size]),self.similar_weight)
        # tmp=tf.reshape(tmp,shape=[-1,3,self.hidden_size])
        # self.middle_out=tf.reshape(tf.matmul(tmp, tf.reshape(self.query_passage_concated,
        #                                                      shape=[-1, self.hidden_size , 1])), shape=[-1, 3])

        alternatives_states=[self.get_h(self.alternatives_state_0),
                                  self.get_h(self.alternatives_state_1),
                                  self.get_h(self.alternatives_state_2)]
        concated_list=[]

        for alternatives_state in alternatives_states:
            concated_list.append(tf.concat([self.query_passage_concated,alternatives_state],axis=1))

        complete_deal=tf.reshape(tf.concat(concated_list,axis=1),shape=[-1,self.hidden_size*2])

        for idx, (weight, bias) in enumerate(self.auto_NN['output_layer']):
            if idx%2==0:
                complete_deal = tf.nn.elu(tf.matmul(complete_deal, weight) + bias)
            else:
                complete_deal = tf.nn.dropout(tf.nn.elu(tf.matmul(complete_deal, weight) + bias),keep_prob=self.keep_pro)

        self.middle_out=tf.reshape(complete_deal,shape=[-1,3])



    def get_h(self, state):
        c, h = state[-1]
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def compute_attention(self, state, outputs, length):
        return tf.reshape(
            tf.matmul(
                tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            tf.matmul(
                                outputs, tf.reshape(
                                    tf.matmul(
                                        state, self.attention_weight), shape=[-1, self.hidden_size, 1])),
                            [-1, self.max_len]) * tf.sequence_mask(
                            length, self.max_len, dtype=tf.float32)), shape=[-1, 1, self.max_len]), outputs),
            shape=[-1, self.hidden_size])

    def computer_loss(self):
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.05)(self.similar_weight))
        self.loss_l2=tf.add_n(tf.get_collection("losses"))

        tf.add_to_collection("losses",
                             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,
                                                                                           logits=self.middle_out)))
        self.loss=tf.add_n(tf.get_collection("losses"))

    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr,0.01,5))
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables),global_step=self.global_step)


class MANM_Model:

    def __init__(self,config):
        self.hidden_size=config['hidden_size']
        self.num_layers=config['num_layers']
        self.word_dim=config['word_dim']
        self.voc_len=config['voc_len']

        self.max_len_passage=config['max_len_passage']
        self.max_len_query = config['max_len_query']

        self.feature_extract_layer_num=3
        self.output_layer_num=3
        self.global_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 800, 0.96, staircase=True)
        self.max_grad_norm=config['max_grad_norm']
        self.auto_NN={}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.forward_step1()
        self.forward_step2()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro=tf.placeholder(dtype=tf.float32,name='keep_pro')

        self.query_input=tf.placeholder(dtype=tf.int32,shape=[None,None],name='query_input')
        self.passage_input=tf.placeholder(dtype=tf.int32,shape=[None,None],name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input=tf.placeholder(dtype=tf.int32,shape=[None],name='y_input')
        self.alternatives_input=tf.placeholder(dtype=tf.int32,shape=[None,3,None],name='alternatives_input')
        self.alternatives_len_input=tf.placeholder(dtype=tf.int32,shape=[None,3],name='alternatives_len_input')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.2, 0.2)
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding=tf.get_variable(dtype=tf.float32,
                                                initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                        dtype=tf.float32),name='word_embedding',trainable=True)
            print('loaded')
        with tf.variable_scope('LSTM_encoder',reuse=None):

            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell=tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*2,self.hidden_size*2],initializer=self.initializer,name='attention_weight')
        with tf.variable_scope('feature_extract'):
            self.generate_NN(self.hidden_size*4,self.hidden_size,self.feature_extract_layer_num,'feature_extract')
        with tf.variable_scope('similar'):
            self.similar_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*2,self.hidden_size*2],name='similar_weight')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*2,1, self.output_layer_num,'output_layer')

    def generate_NN(self,input_size,output_size,layers,name):
        auto_NN_weights=[]
        auto_NN_biases=[]
        for i in range(layers):
            if i==layers-1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32,shape=[input_size,output_size],
                                                       name='auto_NN_weight_'+name+'_'+str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32,shape=[output_size],
                                                      name='auto_NN_bias_'+name+'_'+str(i),initializer=self.initializer))
                # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0000001)(auto_NN_weights[i]))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size , input_size ],
                                                       name='auto_NN_weight_' +name+'_'+ str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size ],
                                                      name='auto_NN_bias_' + name+'_'+str(i),initializer=self.initializer))
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.005)(auto_NN_weights[i]))
        self.auto_NN[name]=zip(auto_NN_weights,auto_NN_biases)

    def forward_step0(self):
        #这一部分主要是lstm编码过程
        query_emb=tf.nn.embedding_lookup(self.word_embedding,self.query_input)
        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)
        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:,0,:])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])
        with tf.variable_scope('query_encoder'):
            self.query_output, self.query_state = tf.nn.bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                                  query_emb, sequence_length=self.query_len_input, dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            self.passage_output,self.passage_state=tf.nn.bidirectional_dynamic_rnn(self.passage_cell_fw, self.passage_cell_bw,
                                                                                   passage_emb, sequence_length=self.passage_len_input, dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, self.alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb_0,sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, self.alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb_1,sequence_length=self.alternatives_len_input[:,1],dtype=tf.float32)
            alternatives_output_2, self.alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb_2,sequence_length=self.alternatives_len_input[:,2],dtype=tf.float32)

    def forward_step1(self):
        #这一部分计算attention后并且提取特征后的结果
        self.query_passage_concated=self.word_level_attention(tf.concat(self.query_output,2),tf.concat(self.passage_output,2),
                                                              self.query_len_input,self.passage_len_input)
        for idx, (weight, bias) in enumerate(self.auto_NN['feature_extract']):
            if idx%2==0:
                self.query_passage_concated = tf.nn.dropout(tf.nn.elu(tf.matmul(self.query_passage_concated,weight)+bias),
                                                            keep_prob=self.keep_pro)
            else:
                self.query_passage_concated = tf.nn.elu(tf.matmul(self.query_passage_concated, weight) + bias)

    def word_level_attention(self,output1,output2,len1,len2):

        output1_list=[]
        for idx in range(self.max_len_query):
            output=output1[:,idx,:]
            output1_list.append(self.compute_attention(output,output2,len2,self.max_len_passage))
        output1_middle=tf.concat(output1_list,axis=1)*\
                       tf.sequence_mask(len1, self.max_len_query*self.hidden_size*2, dtype=tf.float32)

        output2_list = []
        for idx in range(self.max_len_passage):
            output = output2[:,idx,:]
            output2_list.append(self.compute_attention(output, output1, len1,self.max_len_query))
        output2_middle = tf.concat(output2_list, axis=1) * \
                         tf.sequence_mask(len2, self.max_len_passage*self.hidden_size*2, dtype=tf.float32)

        output1_final=tf.reshape(tf.reduce_sum(tf.reshape(output1_middle,
                                                          shape=[-1,self.max_len_query,self.hidden_size*2]),axis=1),
                                 shape=[-1,self.hidden_size*2])

        output2_final = tf.reshape(tf.reduce_sum(tf.reshape(output2_middle,
                                                            shape=[-1, self.max_len_passage, self.hidden_size*2]), axis=1),
                                   shape=[-1, self.hidden_size*2])
        return tf.concat([output1_final,output2_final],axis=1)


    def forward_step2(self):
        #这一部分收尾工作，计算三个答案的具体的值

        alternatives_states=[self.get_h(self.alternatives_state_0),
                                  self.get_h(self.alternatives_state_1),
                                  self.get_h(self.alternatives_state_2)]
        concated_list=[]

        for alternatives_state in alternatives_states:
            concated_list.append(tf.concat([self.query_passage_concated,alternatives_state],axis=1))

        complete_deal=tf.reshape(tf.concat(concated_list,axis=1),shape=[-1,self.hidden_size*2])

        for idx, (weight, bias) in enumerate(self.auto_NN['output_layer']):
            if idx%2==0:
                complete_deal = tf.nn.elu(tf.matmul(complete_deal, weight) + bias)
            else:
                complete_deal = tf.nn.dropout(tf.nn.elu(tf.matmul(complete_deal, weight) + bias),keep_prob=self.keep_pro)

        self.middle_out=tf.reshape(complete_deal,shape=[-1,3])



    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def compute_attention(self, state, outputs, length,max_len):
        return tf.reshape(
            tf.matmul(
                tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            tf.matmul(
                                outputs, tf.reshape(
                                    tf.matmul(
                                        state, self.attention_weight), shape=[-1, self.hidden_size*2, 1])),
                            [-1, max_len]) * tf.sequence_mask(
                            length, max_len, dtype=tf.float32)), shape=[-1, 1, max_len]), outputs),
            shape=[-1, self.hidden_size*2])

    def computer_loss(self):
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.05)(self.similar_weight))
        self.loss_l2=tf.add_n(tf.get_collection("losses"))

        tf.add_to_collection("losses",
                             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,
                                                                                           logits=self.middle_out)))
        self.loss=tf.add_n(tf.get_collection("losses"))

    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr,0.01,5))
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables),global_step=self.global_step)


class MANM_2_Model:
    def __init__(self,config):
        self.hidden_size=config['hidden_size']
        self.num_layers=config['num_layers']
        self.word_dim=config['word_dim']
        self.voc_len=config['voc_len']

        self.max_len_passage=config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives=config['max_len_alternatives']
        self.feature_extract_layer_num=3
        self.output_layer_num=5
        self.global_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 800, 0.96, staircase=True)

        self.max_grad_norm=config['max_grad_norm']
        self.auto_NN={}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro=tf.placeholder(dtype=tf.float32,name='keep_pro')

        self.query_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_query],name='query_input')
        self.passage_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_passage],name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input=tf.placeholder(dtype=tf.int32,shape=[None],name='y_input')
        self.alternatives_input=tf.placeholder(dtype=tf.int32,shape=[None,3,self.max_len_alternatives],name='alternatives_input')
        self.alternatives_len_input=tf.placeholder(dtype=tf.int32,shape=[None,3],name='alternatives_len_input')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding=tf.get_variable(dtype=tf.float32,
                                                initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                        dtype=tf.float32),name='word_embedding',trainable=True)
            print('loaded')
        with tf.variable_scope('LSTM_encoder',reuse=None):

            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell=tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*3,self.hidden_size*2],initializer=self.initializer,name='attention_weight')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*5,1, self.output_layer_num,'output_layer')

    def generate_NN(self,input_size,output_size,layers,name):
        auto_NN_weights=[]
        auto_NN_biases=[]
        for i in range(layers):
            if i==layers-1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32,shape=[input_size,output_size],
                                                       name='auto_NN_weight_'+name+'_'+str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32,shape=[output_size],
                                                      name='auto_NN_bias_'+name+'_'+str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size , input_size ],
                                                       name='auto_NN_weight_' +name+'_'+ str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size ],
                                                      name='auto_NN_bias_' + name+'_'+str(i),initializer=self.initializer))
                # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(auto_NN_weights[i]))
        self.auto_NN[name]=zip(auto_NN_weights,auto_NN_biases)

    def forward_step0(self):
        #这一部分主要是lstm编码过程
        query_emb=tf.nn.embedding_lookup(self.word_embedding,self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb=[alternatives_emb_0,alternatives_emb_1,alternatives_emb_2]

        with tf.variable_scope('query_encoder'):
            query_output, query_state = tf.nn.bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                                  query_emb, sequence_length=self.query_len_input, dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output,passage_state=tf.nn.bidirectional_dynamic_rnn(self.passage_cell_fw, self.passage_cell_bw,
                                                                                   passage_emb, sequence_length=self.passage_len_input, dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                  sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                sequence_length=self.alternatives_len_input[:, 1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                  sequence_length=self.alternatives_len_input[:, 2],
                                                                  dtype=tf.float32)

        alternatives_state=[self.get_h(alternatives_state_0),
                            self.get_h(alternatives_state_1),
                            self.get_h(alternatives_state_2)]

        self.forward_step1(alternatives_state, query_output, passage_output)


    def forward_step1(self,alternatives_state,query_output,passage_output):

        query_passage_concated=self.word_level_attention(alternatives_state,tf.concat(query_output,2),tf.concat(passage_output,2),
                                                              self.query_len_input,self.passage_len_input)
        self.forward_step2(alternatives_state,query_passage_concated)


    def forward_step2(self,alternatives_state,query_passage_concated):
        alternatives_state=tf.reshape(tf.concat(alternatives_state,axis=1),shape=[-1,self.hidden_size])
        complete_deal = tf.concat([query_passage_concated, alternatives_state], axis=1)

        for idx, (weight, bias) in enumerate(self.auto_NN['output_layer']):
            if idx % 2 == 0:
                complete_deal = tf.nn.elu(tf.matmul(complete_deal, weight) + bias)
            else:
                complete_deal = tf.nn.dropout(tf.nn.elu(tf.matmul(complete_deal, weight) + bias),
                                              keep_prob=self.keep_pro)

        self.middle_out = tf.reshape(complete_deal, shape=[-1, 3])

    def word_level_attention(self,alternatives_state,output1,output2,len1,len2):

        len1 = tf.reshape(len1, shape=[-1, 1])
        len1 = tf.reshape(
            tf.concat([len1, len1, len1], axis=1),
            shape=[-1])

        len2 = tf.reshape(len2, shape=[-1, 1])
        len2 = tf.reshape(
            tf.concat([len2, len2, len2], axis=1),
            shape=[-1])

        output1_list=[]

        output1_concat=tf.reshape(tf.concat([output1,output1,output1],axis=1),[-1,self.max_len_query,self.hidden_size*2])
        output2_concat=tf.reshape(tf.concat([output2,output2,output2],axis=1),[-1,self.max_len_passage,self.hidden_size*2])

        for idx in range(self.max_len_query):
            output=tf.concat([output1[:, idx, :],alternatives_state[0],
                              output1[:, idx, :], alternatives_state[1],
                              output1[:, idx, :], alternatives_state[2]],axis=1)
            output1_list.append(self.compute_attention(output,output2_concat,len2,self.max_len_passage))

        output1_middle=tf.concat(output1_list,axis=1)*\
                       tf.sequence_mask(len1, self.max_len_query*self.hidden_size*2, dtype=tf.float32)

        output1_final=tf.reshape(
            tf.reduce_sum(tf.reshape(output1_middle, shape=[-1, self.max_len_query, self.hidden_size * 2]), axis=1),
            shape=[-1, self.hidden_size * 2])

        output2_list = []
        for idx in range(self.max_len_passage):
            output =tf.concat([output2[:, idx, :],alternatives_state[0],
                               output2[:, idx, :], alternatives_state[1],
                               output2[:, idx, :], alternatives_state[2]],axis=1)
            output2_list.append(self.compute_attention(output, output1_concat, len1,self.max_len_query))
        output2_middle = tf.concat(output2_list, axis=1) * \
                         tf.sequence_mask(len2, self.max_len_passage*self.hidden_size*2, dtype=tf.float32)

        output2_final=tf.reshape(
            tf.reduce_sum(tf.reshape(output2_middle,shape=[-1, self.max_len_passage, self.hidden_size*2]), axis=1),
            shape=[-1, self.hidden_size*2])

        return tf.concat([output1_final,output2_final],axis=1)

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def compute_attention(self, state, outputs, length,max_len):
        state=tf.reshape(state,shape=[-1,self.hidden_size*3])

        return tf.reshape(
            tf.matmul(
                tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            tf.matmul(
                                outputs, tf.reshape(
                                    tf.matmul(
                                        state, self.attention_weight), shape=[-1, self.hidden_size*2, 1])),
                            [-1, max_len]) * tf.sequence_mask(
                            length, max_len, dtype=tf.float32)), shape=[-1, 1, max_len]), outputs),
            shape=[-1, self.hidden_size*2])

    def computer_loss(self):

        # tv = tf.trainable_variables()
        # tv.pop(-1)
        # for tensor in tv:
        #     try:
        #         tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(tensor))
        #     except:
        #         pass

        # self.loss_l2=tf.add_n(tf.get_collection("losses"))
        self.loss_l2=tf.constant(1,name='loss_l2')
        tf.add_to_collection("losses",
                             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,
                                                                                           logits=self.middle_out)))
        self.loss=tf.add_n(tf.get_collection("losses"))

    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr,0.01,5))
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables),global_step=self.global_step)

    def val_(self, sess, tensor_list, feed_dict):
        max_len=len(feed_dict['query'])
        scale=1500
        out=[]
        for i in range(10000):
            out.append(sess.run(tensor_list,feed_dict={self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                              self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                              self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                              self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                              self.keep_pro: 1}))
            if (i+1)*scale>=max_len:
                break
        return out


class MANM_3_Model:
    def __init__(self,config):
        self.hidden_size=config['hidden_size']
        self.num_layers=config['num_layers']
        self.word_dim=config['word_dim']
        self.voc_len=config['voc_len']

        self.max_len_passage=config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives=config['max_len_alternatives']
        self.feature_extract_layer_num=3
        self.output_layer_num=5

        self.global_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 800, 0.96, staircase=True)

        self.max_grad_norm=config['max_grad_norm']
        self.auto_NN={}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro=tf.placeholder(dtype=tf.float32,name='keep_pro')

        self.query_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_query],name='query_input')
        self.passage_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_passage],name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input=tf.placeholder(dtype=tf.int32,shape=[None],name='y_input')
        self.alternatives_input=tf.placeholder(dtype=tf.int32,shape=[None,3,self.max_len_alternatives],name='alternatives_input')
        self.alternatives_len_input=tf.placeholder(dtype=tf.int32,shape=[None,3],name='alternatives_len_input')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.25, 0.25)
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding=tf.get_variable(dtype=tf.float32,
                                                initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                        dtype=tf.float32),name='word_embedding',trainable=True)
            print('done')
        with tf.variable_scope('LSTM_encoder',reuse=None):

            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell=tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*3,self.hidden_size*2],initializer=self.initializer,name='attention_weight')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*5,1, self.output_layer_num,'output_layer')

    def generate_NN(self,input_size,output_size,layers,name):
        auto_NN_weights=[]
        auto_NN_biases=[]
        for i in range(layers):
            if i==layers-1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32,shape=[input_size,output_size],
                                                       name='auto_NN_weight_'+name+'_'+str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32,shape=[output_size],
                                                      name='auto_NN_bias_'+name+'_'+str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size , input_size ],
                                                       name='auto_NN_weight_' +name+'_'+ str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size ],
                                                      name='auto_NN_bias_' + name+'_'+str(i),initializer=self.initializer))
                # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(auto_NN_weights[i]))
        self.auto_NN[name]=zip(auto_NN_weights,auto_NN_biases)

    def forward_step0(self):
        #这一部分主要是lstm编码过程
        query_emb=tf.nn.embedding_lookup(self.word_embedding,self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb=[alternatives_emb_0,alternatives_emb_1,alternatives_emb_2]

        with tf.variable_scope('query_encoder'):
            query_output, query_state = tf.nn.bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                                  query_emb, sequence_length=self.query_len_input, dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output,passage_state=tf.nn.bidirectional_dynamic_rnn(self.passage_cell_fw, self.passage_cell_bw,
                                                                                   passage_emb, sequence_length=self.passage_len_input, dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                  sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                sequence_length=self.alternatives_len_input[:, 1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                  sequence_length=self.alternatives_len_input[:, 2],
                                                                  dtype=tf.float32)

        alternatives_state=[self.get_h(alternatives_state_0),
                            self.get_h(alternatives_state_1),
                            self.get_h(alternatives_state_2)]

        tmp0=self.forward_step1(alternatives_state[0], query_output, passage_output)
        tmp1=self.forward_step1(alternatives_state[1], query_output, passage_output)
        tmp2=self.forward_step1(alternatives_state[2], query_output, passage_output)

        self.middle_out=tf.transpose(tf.reshape(self.forward_step2(tf.concat([tmp0,tmp1,tmp2],axis=0)),[3,-1]))


    def forward_step1(self,alternatives_state,query_output,passage_output):
        query_passage_concated=self.word_level_attention(alternatives_state,tf.concat(query_output,2),tf.concat(passage_output,2),
                                                              self.query_len_input,self.passage_len_input)
        concated = tf.concat([query_passage_concated, alternatives_state], axis=1)
        return concated


    def forward_step2(self,concated):
        for idx, (weight, bias) in enumerate(self.auto_NN['output_layer']):
            if idx % 2 == 0:
                concated = tf.nn.elu(tf.matmul(concated, weight) + bias)
            else:
                concated = tf.nn.dropout(tf.nn.elu(tf.matmul(concated, weight) + bias),
                                              keep_prob=self.keep_pro)

        return concated

    def word_level_attention(self,alternatives_state,output1,output2,len1,len2):

        output1_list=[]
        for idx in range(self.max_len_query):
            output=tf.concat([output1[:, idx, :],alternatives_state],axis=1)
            output1_list.append(self.compute_attention(output,output2,len2,self.max_len_passage))

        output1_middle=tf.concat(output1_list,axis=1)*\
                       tf.sequence_mask(len1, self.max_len_query*self.hidden_size*2, dtype=tf.float32)

        output1_final=tf.reshape(
            tf.reduce_mean(tf.reshape(output1_middle, shape=[-1, self.max_len_query, self.hidden_size * 2]), axis=1),
            shape=[-1, self.hidden_size * 2])

        output2_list = []
        for idx in range(self.max_len_passage):
            output =tf.concat([output2[:, idx, :],alternatives_state],axis=1)
            output2_list.append(self.compute_attention(output, output1, len1,self.max_len_query))
        output2_middle = tf.concat(output2_list, axis=1) * \
                         tf.sequence_mask(len2, self.max_len_passage*self.hidden_size*2, dtype=tf.float32)

        output2_final=tf.reshape(
            tf.reduce_mean(tf.reshape(output2_middle,shape=[-1, self.max_len_passage, self.hidden_size*2]), axis=1),
            shape=[-1, self.hidden_size*2])

        return tf.concat([output1_final,output2_final],axis=1)

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def compute_attention(self, state, outputs, length,max_len):
        return tf.reshape(
            tf.matmul(
                tf.reshape(
                    tf.nn.softmax(
                        tf.reshape(
                            tf.matmul(
                                outputs, tf.reshape(
                                    tf.matmul(
                                        state, self.attention_weight), shape=[-1, self.hidden_size*2, 1])),
                            [-1, max_len]) * tf.sequence_mask(
                            length, max_len, dtype=tf.float32)), shape=[-1, 1, max_len]), outputs),
            shape=[-1, self.hidden_size*2])

    def computer_loss(self):

        # tv = tf.trainable_variables()
        #
        # for tensor in tv:
        #
        #     try:
        #         if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'lstm' in tensor.name:
        #             print(tensor)
        #             tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.00005)(tensor))
        #     except:
        #         pass
        #
        # self.loss_l2=tf.add_n(tf.get_collection("losses"))
        self.loss_l2=tf.constant(1)
        tf.add_to_collection("losses",
                             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,
                                                                                           logits=self.middle_out)))
        self.loss=tf.add_n(tf.get_collection("losses"))

    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr,0.01,5))
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables),global_step=self.global_step)

    def val_(self, sess, tensor_list, feed_dict):
        max_len=len(feed_dict['query'])
        scale=1500
        out=[]
        for i in range(10000):
            out.append(sess.run(tensor_list,feed_dict={self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                              self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                              self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                              self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                              self.keep_pro: 1}))
            if (i+1)*scale>=max_len:
                break
        return out

class MANM_4_Model:

    def __init__(self,config):
        self.hidden_size=config['hidden_size']
        self.num_layers=config['num_layers']
        self.word_dim=config['word_dim']
        self.voc_len=config['voc_len']

        self.max_len_passage=config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives=config['max_len_alternatives']
        self.output_layer_num=4

        self.global_step = tf.Variable(0,trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm=config['max_grad_norm']
        self.flag=tf.constant(dtype=tf.int32,value=0,name='flag')
        self.auto_NN={}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro=tf.placeholder(dtype=tf.float32,name='keep_pro')

        self.query_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_query],name='query_input')
        self.passage_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_passage],name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input=tf.placeholder(dtype=tf.int64,shape=[None],name='y_input')
        self.alternatives_input=tf.placeholder(dtype=tf.int32,shape=[None,3,self.max_len_alternatives],name='alternatives_input')
        self.alternatives_len_input=tf.placeholder(dtype=tf.int32,shape=[None,3],name='alternatives_len_input')

        self.whether_train=tf.placeholder(dtype=tf.int32,name='whether_train')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.2, 0.2)
        # self.initializer = tf.random_normal_initializer()
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding=tf.get_variable(dtype=tf.float32,
                                                initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                        dtype=tf.float32),name='word_embedding',trainable=True)
            print('done')
        with tf.variable_scope('LSTM_encoder',reuse=None):

            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell=tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*2,self.hidden_size*3],initializer=self.initializer,name='attention_weight')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*5,1, self.output_layer_num,'output_layer')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_NN(self,input_size,output_size,layers,name):
        auto_NN_weights=[]
        auto_NN_biases=[]
        auto_NN_betas=[]
        auto_NN_scales=[]
        for i in range(layers):
            if i==layers-1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32,shape=[input_size*4,output_size],
                                                       name='auto_NN_weight_'+name+'_'+str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32,shape=[output_size],
                                                      name='auto_NN_bias_'+name+'_'+str(i),initializer=self.initializer))
            elif i==0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size ,  input_size*4 ],
                                                       name='auto_NN_weight_' +name+'_'+ str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size*4 ],
                                                      name='auto_NN_bias_' + name+'_'+str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size*4, input_size * 4],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))

            auto_NN_betas.append(tf.get_variable(dtype=tf.float32, shape=[1],name='auto_NN_beta_' +name+'_'+ str(i),initializer=self.initializer))
            auto_NN_scales.append(tf.get_variable(dtype=tf.float32, shape=[1],name='auto_NN_scale_' + name + '_' + str(i),initializer=self.initializer))
        self.auto_NN[name]=zip(auto_NN_weights,auto_NN_biases,auto_NN_betas,auto_NN_scales)

    def forward_step0(self):
        #这一部分主要是lstm编码过程
        query_emb=tf.nn.embedding_lookup(self.word_embedding,self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb=[alternatives_emb_0,alternatives_emb_1,alternatives_emb_2]

        with tf.variable_scope('query_encoder'):
            query_output, query_state = tf.nn.bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                                  query_emb, sequence_length=self.query_len_input, dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output,passage_state=tf.nn.bidirectional_dynamic_rnn(self.passage_cell_fw, self.passage_cell_bw,
                                                                                   passage_emb, sequence_length=self.passage_len_input, dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                  sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                sequence_length=self.alternatives_len_input[:, 1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                  sequence_length=self.alternatives_len_input[:, 2],
                                                                  dtype=tf.float32)

        alternatives_state=[self.get_h(alternatives_state_0),
                            self.get_h(alternatives_state_1),
                            self.get_h(alternatives_state_2)]
        query_output=tf.concat(query_output,axis=2)
        passage_output=tf.concat(passage_output,axis=2)
        tmp0=self.forward_step1(alternatives_state[0], query_output, passage_output)
        tmp1=self.forward_step1(alternatives_state[1], query_output, passage_output)
        tmp2=self.forward_step1(alternatives_state[2], query_output, passage_output)
        # self.tmp=tf.concat([tmp0,tmp1,tmp2],axis=0)
        self.middle_out=tf.transpose(tf.reshape(self.forward_step2(tf.concat([tmp0,tmp1,tmp2],axis=0)),[3,-1]))


    def forward_step1(self,alternatives_state,query_output,passage_output):
        query_attentioned=self.compute_attention(alternatives_state,query_output,passage_output,self.query_len_input,
                                                 self.passage_len_input,self.max_len_query,self.max_len_passage)
        self.tmp_tmp=query_attentioned
        passage_attentioned=self.compute_attention(alternatives_state,passage_output,query_output,self.passage_len_input,
                                                   self.query_len_input,self.max_len_passage,self.max_len_query)
        concated = tf.concat([query_attentioned,passage_attentioned, alternatives_state], axis=1)
        return concated

#小于0是训练 大于0是预测
    def forward_step2(self,concated):
        for idx, (weight, bias,beta,scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))
            concated=tf.matmul(concated, weight) + bias
            # batch_mean, batch_var = tf.nn.moments(tmp, [0])
            # tmp = tf.nn.batch_normalization(tmp, batch_mean, batch_var, beta, scale, 1e-3)
            concated=tf.nn.elu(concated)
            if idx==self.output_layer_num-1:
                concated=concated
            else:
                concated = tf.nn.dropout(concated,keep_prob=self.keep_pro)
        return concated

    def doto_the_ctrl(self,weight,bias,beta,scale,flag):
        if flag:
            return [weight,bias,beta,scale]
        else:
            return [self.ema.average(weight),self.ema.average(bias),self.ema.average(beta),self.ema.average(scale)]

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def compute_attention(self,alternatives_state,outputs1,outputs2,len1,len2,max_len1,max_len2):
        alternatives_state_list=[]
        for i in range(max_len2):
            alternatives_state_list.append(alternatives_state)
        al=tf.reshape(tf.concat(alternatives_state_list,axis=1),[-1,self.hidden_size])

        outputs2=tf.concat([tf.reshape(outputs2,[-1,self.hidden_size*2]),al],axis=1)
        outputs2=tf.transpose(outputs2)
        tmp=tf.matmul(self.attention_weight,outputs2)
        tmp=tf.transpose(tf.reshape(tf.transpose(tmp),[-1,max_len2,self.hidden_size*2]),perm=[0, 2, 1])
        tmp=tf.matmul(outputs1,tmp)

        mask=tf.reshape(tf.sequence_mask(len2,max_len2, dtype=tf.float32),[-1,max_len2,1])
        tmp=tf.matmul(tmp,mask)

        tmp=tf.reshape(tmp,[-1,max_len1])*tf.sequence_mask(len1,max_len1, dtype=tf.float32)
        tmp=tf.reshape(tmp,[-1,1,max_len1])
        return tf.reshape(tf.matmul(tmp,outputs1),[-1,self.hidden_size*2])

    def computer_loss(self):

        self.tv = tf.trainable_variables()
        # for tensor in self.tv:
        #     try:
        #         if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
        #         # if 'embedding' in tensor.name:
        #             print(tensor)
        #             tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.000000001)(tensor))
        #     except:
        #         pass
        # self.loss_l2=tf.add_n(tf.get_collection("losses"))
        self.loss_l2=tf.constant(1)
        # tf.cast(tf.equal(tf.argmax(self.middle_out, axis=1), tf.reshape(self.y_input, [-1])), dtype=tf.float32) *
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,logits=self.middle_out))
        tf.add_to_collection("losses",loss)
        self.loss=tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr,0.01,5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv),global_step=self.global_step),self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict,flag):
        max_len=len(feed_dict['query'])
        scale=10000
        out=[]
        for i in range(10000):
            out.append(sess.run(tensor_list,feed_dict={self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                              self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                              self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                              self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                              self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                              self.keep_pro: 1,
                            self.whether_train:-1}))
            if (i+1)*scale>=max_len:
                break
        return out


class MANM_5_Model:

    def __init__(self,config):
        self.hidden_size=config['hidden_size']
        self.num_layers=config['num_layers']
        self.word_dim=config['word_dim']
        self.voc_len=config['voc_len']

        self.max_len_passage=config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives=config['max_len_alternatives']
        self.feature_extract_layer_num=3
        self.output_layer_num=3

        self.global_step = tf.Variable(0,trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 800, 0.96, staircase=True)

        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.max_grad_norm=config['max_grad_norm']
        self.auto_NN={}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro=tf.placeholder(dtype=tf.float32,name='keep_pro')

        self.query_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_query],name='query_input')
        self.passage_input=tf.placeholder(dtype=tf.int32,shape=[None,self.max_len_passage],name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input=tf.placeholder(dtype=tf.int32,shape=[None],name='y_input')
        self.alternatives_input=tf.placeholder(dtype=tf.int32,shape=[None,3,self.max_len_alternatives],name='alternatives_input')
        self.alternatives_len_input=tf.placeholder(dtype=tf.int32,shape=[None,3],name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')
    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.25, 0.25)
        # self.initializer = tf.random_normal_initializer()
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding=tf.get_variable(dtype=tf.float32,
                                                initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                        dtype=tf.float32),name='word_embedding',trainable=True)
            print('done')
        with tf.variable_scope('LSTM_encoder',reuse=None):

            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell=tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*2,self.hidden_size*3],initializer=self.initializer,name='attention_weight')
            self.attention_weight_with_al=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size,self.hidden_size*2],initializer=self.initializer,name='attention_weight_with_al')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*5,1, self.output_layer_num,'output_layer')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_NN(self,input_size,output_size,layers,name):
        auto_NN_weights=[]
        auto_NN_biases=[]
        auto_NN_betas=[]
        auto_NN_scales=[]
        for i in range(layers):
            if i==layers-1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32,shape=[input_size,output_size],
                                                       name='auto_NN_weight_'+name+'_'+str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32,shape=[output_size],
                                                      name='auto_NN_bias_'+name+'_'+str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size , input_size ],
                                                       name='auto_NN_weight_' +name+'_'+ str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size ],
                                                      name='auto_NN_bias_' + name+'_'+str(i),initializer=self.initializer))

            auto_NN_betas.append(tf.get_variable(dtype=tf.float32, shape=[1],name='auto_NN_beta_' +name+'_'+ str(i),initializer=self.initializer))
            auto_NN_scales.append(tf.get_variable(dtype=tf.float32, shape=[1],name='auto_NN_scale_' + name + '_' + str(i),initializer=self.initializer))
        self.auto_NN[name]=zip(auto_NN_weights,auto_NN_biases,auto_NN_betas,auto_NN_scales)

    def forward_step0(self):
        #这一部分主要是lstm编码过程
        query_emb=tf.nn.embedding_lookup(self.word_embedding,self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb=[alternatives_emb_0,alternatives_emb_1,alternatives_emb_2]

        with tf.variable_scope('query_encoder'):
            query_output, query_state = tf.nn.bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                                  query_emb, sequence_length=self.query_len_input, dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output,passage_state=tf.nn.bidirectional_dynamic_rnn(self.passage_cell_fw, self.passage_cell_bw,
                                                                                   passage_emb, sequence_length=self.passage_len_input, dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                  sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                sequence_length=self.alternatives_len_input[:, 1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                  sequence_length=self.alternatives_len_input[:, 2],
                                                                  dtype=tf.float32)

        alternatives_state=[self.get_h(alternatives_state_0),
                            self.get_h(alternatives_state_1),
                            self.get_h(alternatives_state_2)]
        query_output=tf.concat(query_output,axis=2)
        passage_output=tf.concat(passage_output,axis=2)
        tmp0=self.forward_step1(alternatives_state[0], query_output, passage_output)
        tmp1=self.forward_step1(alternatives_state[1], query_output, passage_output)
        tmp2=self.forward_step1(alternatives_state[2], query_output, passage_output)
        self.tmp=tf.concat([tmp0,tmp1,tmp2],axis=0)
        self.middle_out=tf.transpose(tf.reshape(self.forward_step2(tf.concat([tmp0,tmp1,tmp2],axis=0)),[3,-1]))


    def forward_step1(self,alternatives_state,query_output,passage_output):
        query_attentioned=self.compute_attention(alternatives_state,query_output,passage_output,self.query_len_input,
                                                 self.passage_len_input,self.max_len_query,self.max_len_passage)
        passage_attentioned=self.compute_attention(alternatives_state,passage_output,query_output,self.passage_len_input,
                                                   self.query_len_input,self.max_len_passage,self.max_len_query)
        concated = tf.concat([query_attentioned,passage_attentioned, alternatives_state], axis=1)
        return concated

    def forward_step2(self,concated):
        for idx, (weight, bias,beta,scale) in enumerate(self.auto_NN['output_layer']):
            weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
                                              lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
                                              lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))
            tmp=tf.nn.elu(tf.matmul(concated, weight) + bias)
            batch_mean, batch_var = tf.nn.moments(tmp, [0])
            BN = tf.nn.batch_normalization(tmp, batch_mean, batch_var, beta, scale, 1e-3)
            if idx==self.output_layer_num-1:
                concated=BN
            else:
                concated = tf.nn.dropout(BN,keep_prob=self.keep_pro)
        return concated

    def doto_the_ctrl(self,weight,bias,beta,scale,flag):
        if flag:
            return [weight,bias,beta,scale]
        else:
            return [self.ema.average(weight),self.ema.average(bias),self.ema.average(beta),self.ema.average(scale)]

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def compute_attention(self,alternatives_state,outputs1,outputs2,len1,len2,max_len1,max_len2):

        alternatives_state_list = []
        for i in range(max_len2):
            alternatives_state_list.append(alternatives_state)

        al = tf.reshape(tf.concat(alternatives_state_list, axis=1), [-1, self.hidden_size])

        mask = tf.reshape(tf.sequence_mask(len2, max_len2, dtype=tf.float32), [-1, max_len2, 1])
        mask = self.sub_compute_attention(alternatives_state, outputs2, len2, max_len2)*mask

        outputs2 = tf.concat([tf.reshape(outputs2, [-1, self.hidden_size * 2]), al], axis=1)
        outputs2=tf.transpose(outputs2)
        tmp=tf.matmul(self.attention_weight,outputs2)
        tmp=tf.transpose(tf.reshape(tf.transpose(tmp),[-1,max_len2,self.hidden_size*2]),perm=[0, 2, 1])
        tmp=tf.matmul(outputs1,tmp)

        tmp=tf.matmul(tmp,mask)

        tmp=tf.reshape(tmp,[-1,max_len1])*tf.sequence_mask(len1,max_len1, dtype=tf.float32)
        tmp=tf.reshape(tmp,[-1,1,max_len1])
        return tf.reshape(tf.matmul(tmp,outputs1),[-1,self.hidden_size*2])

    def sub_compute_attention(self, state, outputs, length, max_len):
        # outputs=tf.reshape(outputs,[-1,self.hidden_size*2])
        return tf.reshape(tf.nn.relu(
                        tf.reshape(
                            tf.matmul(
                                outputs, tf.reshape(
                                    tf.matmul(
                                        state, self.attention_weight_with_al), shape=[-1, self.hidden_size*2, 1])),
                            [-1, max_len]) * tf.sequence_mask(
                            length, max_len, dtype=tf.float32)),[-1,max_len,1])

    def computer_loss(self):
        # tv = tf.trainable_variables()
        # for tensor in tv:
        #     try:
        #         if not 'bias' in tensor.name and not 'embedding' in tensor.name:
        #             print(tensor)
        #             tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(tensor))
        #     except:
        #         pass
        # self.loss_l2=tf.add_n(tf.get_collection("losses"))
        self.loss_l2=tf.constant(1)
        tf.add_to_collection("losses",
                             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,
                                                                                           logits=self.middle_out)))
        self.loss=tf.add_n(tf.get_collection("losses"))

    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr,0.01,5))
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables),global_step=self.global_step)

    def val_(self, sess, tensor_list, feed_dict,flag):
        max_len=len(feed_dict['query'])
        scale=10000
        out=[]
        for i in range(10000):
            out.append(sess.run(tensor_list,feed_dict={self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                                self.keep_pro: 1,
                                self.whether_train:-1}))
            if (i+1)*scale>=max_len:
                break
        return out


class MANM_6_Model:
    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.feature_extract_layer_num = 3
        self.output_layer_num = 7

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 400, 0.96, staircase=True)

        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.max_grad_norm = config['max_grad_norm']
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage], name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-2, 2)
        # self.initializer = tf.random_normal_initializer()
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding = tf.get_variable(dtype=tf.float32,
                                                  initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                          dtype=tf.float32), name='word_embedding',
                                                  trainable=True)
            print('done')
        with tf.variable_scope('LSTM_encoder', reuse=None):
            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight = tf.get_variable(dtype=tf.float32,
                                                    shape=[self.hidden_size * 2, self.hidden_size * 3],
                                                    initializer=self.initializer, name='attention_weight')
            self.attention_weight_with_al = tf.get_variable(dtype=tf.float32,
                                                            shape=[self.hidden_size, self.hidden_size * 2],
                                                            initializer=self.initializer,
                                                            name='attention_weight_with_al')
            self.attention_weight_word_similar=tf.get_variable(dtype=tf.float32,
                                                    shape=[ self.word_dim,self.word_dim],
                                                    initializer=self.initializer, name='attention_weight_word_similar')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*5, 1, self.output_layer_num, 'output_layer')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, output_size],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))

            auto_NN_betas.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),
                                initializer=self.initializer))
            auto_NN_scales.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),
                                initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)

    def forward_step0(self):
        # 这一部分主要是lstm编码过程

        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb = [alternatives_emb_0, alternatives_emb_1, alternatives_emb_2]

        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                      sequence_length=self.alternatives_len_input[:, 0],
                                                                      dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                      sequence_length=self.alternatives_len_input[:, 1],
                                                                      dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                      sequence_length=self.alternatives_len_input[:, 2],
                                                                      dtype=tf.float32)

        alternatives_state = [self.get_h(alternatives_state_0),
                              self.get_h(alternatives_state_1),
                              self.get_h(alternatives_state_2)]

        mask = tf.reshape(tf.sequence_mask(self.max_len_query, self.max_len_query, dtype=tf.float32),
                          [-1, self.max_len_query, 1])
        passage_emb = self.compute_attention_word(tf.reduce_sum(query_emb * mask, axis=1), passage_emb,
                                                self.passage_len_input, self.max_len_passage)

        with tf.variable_scope('query_encoder'):
            query_output, query_state = tf.nn.bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                                  query_emb, sequence_length=self.query_len_input, dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output,passage_state=tf.nn.bidirectional_dynamic_rnn(self.passage_cell_fw, self.passage_cell_bw,
                                                                                   passage_emb, sequence_length=self.passage_len_input, dtype=tf.float32)


        query_output=tf.concat(query_output,2)
        passage_output=tf.concat(passage_output,2)

        tmp0 = self.forward_step1(alternatives_state[0], query_output, passage_output)
        tmp1 = self.forward_step1(alternatives_state[1], query_output, passage_output)
        tmp2 = self.forward_step1(alternatives_state[2], query_output, passage_output)

        self.tmp = tf.concat([tmp0, tmp1, tmp2], axis=0)
        self.middle_out = tf.transpose(tf.reshape(self.forward_step2(tf.concat([tmp0, tmp1, tmp2], axis=0)), [3, -1]))

    def compute_attention_word(self, reduce_sum,embedding,len,max_len):
        mask = tf.reshape(tf.sequence_mask(len, max_len, dtype=tf.float32), [-1, max_len, 1])
        tmp=tf.matmul(reduce_sum,self.attention_weight_word_similar)
        tmp=tf.reshape(tmp,[-1,self.word_dim,1])
        tmp=tf.matmul(embedding-1,tmp)
        tmp=embedding*tmp*mask
        return tmp


    def forward_step1(self, alternatives_state, query_output, passage_output):
        query_attentioned = self.compute_attention_state(alternatives_state, query_output, passage_output,
                                                         self.query_len_input,
                                                         self.passage_len_input, self.max_len_query, self.max_len_passage)
        passage_attentioned = self.compute_attention_state(alternatives_state, passage_output, query_output,
                                                           self.passage_len_input,
                                                           self.query_len_input, self.max_len_passage, self.max_len_query)
        concated = tf.concat([query_attentioned, passage_attentioned, alternatives_state], axis=1)
        return concated

    def forward_step2(self, concated):
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            weight, bias, beta, scale = tf.cond(self.whether_train < self.flag,
                                                lambda: self.doto_the_ctrl(weight, bias, beta, scale, True),
                                                lambda: self.doto_the_ctrl(weight, bias, beta, scale, False))
            tmp = tf.nn.elu(tf.matmul(concated, weight) + bias)
            batch_mean, batch_var = tf.nn.moments(tmp, [0])
            BN = tf.nn.batch_normalization(tmp, batch_mean, batch_var, beta, scale, 1e-3)
            if idx == self.output_layer_num - 1:
                concated = BN
            else:
                concated = tf.nn.dropout(BN, keep_prob=self.keep_pro)
        return concated

    def doto_the_ctrl(self, weight, bias, beta, scale, flag):
        if flag:
            return [weight, bias, beta, scale]
        else:
            return [self.ema.average(weight), self.ema.average(bias), self.ema.average(beta), self.ema.average(scale)]

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def compute_attention_state(self,alternatives_state,outputs1,outputs2,len1,len2,max_len1,max_len2):

        alternatives_state_list = []
        for i in range(max_len2):
            alternatives_state_list.append(alternatives_state)

        al = tf.reshape(tf.concat(alternatives_state_list, axis=1), [-1, self.hidden_size])

        mask = tf.reshape(tf.sequence_mask(len2, max_len2, dtype=tf.float32), [-1, max_len2, 1])
        mask = self.sub_compute_attention_state(alternatives_state, outputs2, len2, max_len2)*mask

        outputs2 = tf.concat([tf.reshape(outputs2, [-1, self.hidden_size * 2]), al], axis=1)
        outputs2=tf.transpose(outputs2)
        tmp=tf.matmul(self.attention_weight,outputs2)
        tmp=tf.transpose(tf.reshape(tf.transpose(tmp),[-1,max_len2,self.hidden_size*2]),perm=[0, 2, 1])
        tmp=tf.matmul(outputs1,tmp)

        tmp=tf.matmul(tmp,mask)

        tmp=tf.reshape(tmp,[-1,max_len1])*tf.sequence_mask(len1,max_len1, dtype=tf.float32)
        tmp=tf.reshape(tmp,[-1,1,max_len1])
        return tf.reshape(tf.matmul(tmp,outputs1),[-1,self.hidden_size*2])

    def sub_compute_attention_state(self, state, outputs, length, max_len):
        # outputs=tf.reshape(outputs,[-1,self.hidden_size*2])
        return tf.reshape(tf.nn.relu(
                        tf.reshape(
                            tf.matmul(
                                outputs, tf.reshape(
                                    tf.matmul(
                                        state, self.attention_weight_with_al), shape=[-1, self.hidden_size*2, 1])),
                            [-1, max_len]) * tf.sequence_mask(
                            length, max_len, dtype=tf.float32)),[-1,max_len,1])

    def computer_loss(self):
        tv = tf.trainable_variables()
        for tensor in tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1)
        tf.add_to_collection("losses",
                             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input,
                                                                                           logits=self.middle_out)))
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.01, 5))
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out


class MANM_7_Model:

    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.output_layer_num = 6

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm = config['max_grad_norm']
        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage],
                                            name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding = tf.get_variable(dtype=tf.float32,
                                                  initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),
                                                                          dtype=tf.float32), name='word_embedding',
                                                  trainable=True)
            print('done')
        with tf.variable_scope('LSTM_encoder', reuse=None):
            self.query_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.query_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.passage_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.passage_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.alternatives_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight = tf.get_variable(dtype=tf.float32,shape=[self.hidden_size * 2, self.hidden_size * 3],initializer=self.initializer, name='attention_weight')
            self.attention_weight_word=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size * 2, self.word_dim ],initializer=self.initializer, name='attention_weight_word')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size * 5, 1, self.output_layer_num, 'output_layer')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, output_size],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            elif i == 0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_betas.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_scales.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)
        self.tmp_tmp=tf.constant(1)

    def forward_step0(self):
        # 这一部分主要是lstm编码过程
        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb = [alternatives_emb_0, alternatives_emb_1, alternatives_emb_2]


        with tf.variable_scope('query_encoder'):
            query_output, query_state = bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,query_emb,sequence_length=self.query_len_input,dtype=tf.float32)
        # query_state = self.get_bi_h(query_state)
        # passage_emb = self.compute_word_attention(passage_emb, query_state)

        with tf.variable_scope('passage_encoder'):
            passage_output, passage_state = bidirectional_dynamic_rnn(self.passage_cell_fw,self.passage_cell_bw,passage_emb,sequence_length=self.passage_len_input,dtype=tf.float32)
        # passage_state = self.get_bi_h(passage_state)

        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],sequence_length=self.alternatives_len_input[:,1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],sequence_length=self.alternatives_len_input[:,2],dtype=tf.float32)


        al_list=[self.get_h(alternatives_state_0),self.get_h(alternatives_state_1),self.get_h(alternatives_state_2)]
        query_output=tf.concat(query_output,2)
        passage_output=tf.concat(passage_output,2)
        query_state=self.get_bi_h(query_state)
        passage_state=self.get_bi_h(passage_state)

        self.middle_tensor_list=[]
        for i in range(3):
            # query_state = self.compute_attention(al_list[i], query_output, passage_output, self.query_len_input,self.passage_len_input, self.max_len_query, self.max_len_passage)
            # passage_state = self.compute_attention(al_list[i], passage_output, query_output, self.passage_len_input,self.query_len_input, self.max_len_passage, self.max_len_query)

            tmp=tf.concat([query_state,passage_state,al_list[i]],axis=1)
            self.middle_tensor_list.append(tmp)
        tmp=tf.concat(self.middle_tensor_list,axis=0)
        self.middle_out=tf.transpose(tf.reshape(tf.transpose(self.output_layer_forward(tmp)),[3,-1]))

    def compute_word_attention(self, passage_emb, query_state):
        tmp=tf.matmul(query_state,self.attention_weight_word)
        tmp=tf.reshape(tmp,[-1,self.word_dim,1])
        tmp=tf.matmul(passage_emb, tmp)
        mask=tf.reshape(tf.sequence_mask(self.passage_len_input,self.max_len_passage,dtype=tf.float32),[-1,self.max_len_passage,1])
        tmp=tf.nn.softsign(tmp*mask)
        tmp=tmp*passage_emb#*self.max_len_passage
        return tmp

    def compute_attention(self,alternatives_state,outputs1,outputs2,len1,len2,max_len1,max_len2):
        alternatives_state_list=[]
        for i in range(max_len2):
            alternatives_state_list.append(alternatives_state)
        al=tf.reshape(tf.concat(alternatives_state_list,axis=1),[-1,self.hidden_size])

        outputs2=tf.concat([tf.reshape(outputs2,[-1,self.hidden_size*2]),al],axis=1)
        outputs2=tf.transpose(outputs2)
        tmp=tf.matmul(self.attention_weight,outputs2)
        tmp=tf.transpose(tf.reshape(tf.transpose(tmp),[-1,max_len2,self.hidden_size*2]),perm=[0, 2, 1])
        tmp=tf.matmul(outputs1,tmp)

        mask=tf.reshape(tf.sequence_mask(len2,max_len2, dtype=tf.float32),[-1,max_len2,1])
        tmp=tf.matmul(tmp,mask)

        tmp=tf.reshape(tmp,[-1,max_len1])*tf.sequence_mask(len1,max_len1, dtype=tf.float32)
        tmp=tf.reshape(tmp,[-1,1,max_len1])
        return tf.reshape(tf.matmul(tmp,outputs1),[-1,self.hidden_size*2])


    def output_layer_forward(self,input_tensor):
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))
            input_tensor = tf.matmul(input_tensor, weight) + bias
            self.tmp=weight
            # batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
            # input_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, scale, 1e-3)
            # if idx%2==0:
            #     input_tensor=tf.nn.softsign(input_tensor)
            # else:
            input_tensor = tf.nn.elu(input_tensor)
            if idx == self.output_layer_num - 1:
                out = input_tensor
            else:
                out = tf.nn.dropout(input_tensor, keep_prob=self.keep_pro)
        return out

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def get_bi_h(self,state):
        f,b=state
        return tf.concat([self.get_h(f),self.get_h(b)],axis=1)

    def computer_loss(self):
        self.tv = tf.trainable_variables()
        for tensor in self.tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.005)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1.)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.middle_out))
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.01, 5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv), global_step=self.global_step),
                                 self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out




class MANM_8_Model:

    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.output_layer_num = 6

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm = config['max_grad_norm']
        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage],
                                            name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding = tf.get_variable(dtype=tf.float32,initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),dtype=tf.float32), name='word_embedding',trainable=True)
            print('done')

        with tf.variable_scope('LSTM_encoder', reuse=None):
            self.query_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.query_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.passage_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.passage_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            self.alternatives_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight = tf.get_variable(dtype=tf.float32,shape=[self.hidden_size * 2, self.hidden_size * 3],initializer=self.initializer, name='attention_weight')
            self.attention_weight_word=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*2, self.hidden_size*2 ],initializer=self.initializer, name='attention_weight_word')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size*5, 1, self.output_layer_num, 'output_layer')

        with tf.variable_scope('pos_metrix'):
            self.pos_metrix=tf.get_variable(dtype=tf.float32,initializer=self.generate_pos_vector(100,128).astype(np.float32),name='pos_metrix',trainable=False)
            self.pos_metrix_dy=tf.get_variable(dtype=tf.float32,shape=[100,128],initializer=self.initializer,name='pos_metrix_dy')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())


    def generate_pos_vector(self,pos,dim):
        pos_me=np.zeros((pos,dim))
        for idx in range(pos):
            for dim_idx in range(dim):
                if dim_idx%2==0:
                    pos_me[idx,dim_idx]=math.sin(idx/math.pow(10000,dim_idx/dim))
                else:
                    pos_me[idx, dim_idx] = math.cos(idx / math.pow(10000, (dim_idx-1) / dim))
        return pos_me

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, output_size],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            elif i == 0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_betas.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_scales.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)
        self.tmp_tmp=tf.constant(1)

    def forward_step0(self):
        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)#+self.pos_metrix[:self.max_len_query,:]

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)#+self.pos_metrix[:self.max_len_passage,:]

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])#+self.pos_metrix[:self.max_len_alternatives,:]
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])#+self.pos_metrix[:self.max_len_alternatives,:]
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])#+self.pos_metrix[:self.max_len_alternatives,:]

        # passage_emb=self.compute_word_attention(passage_emb,query_emb)
        #passage_emb = self.self_attention(query_emb, passage_emb, self.query_len_input, self.passage_len_input,self.max_len_query, self.max_len_passage)

        alternatives_emb_0 = self.self_attention(alternatives_emb_0,alternatives_emb_0,
                                                               self.alternatives_len_input[:,0],
                                                               self.alternatives_len_input[:,0],
                                                                self.max_len_alternatives,
                                                               self.max_len_alternatives)
        alternatives_emb_1 = self.self_attention(alternatives_emb_1,alternatives_emb_1,
                                                               self.alternatives_len_input[:,1],
                                                               self.alternatives_len_input[:,1],
                                                               self.max_len_alternatives,
                                                               self.max_len_alternatives)
        alternatives_emb_2 = self.self_attention(alternatives_emb_2,
                                                               alternatives_emb_2,
                                                               self.alternatives_len_input[:,2],
                                                               self.alternatives_len_input[:,2],
                                                               self.max_len_alternatives,
                                                               self.max_len_alternatives)

        alternatives_emb = [alternatives_emb_0, alternatives_emb_1, alternatives_emb_2]

        query_emb = self.self_attention(query_emb, query_emb, self.query_len_input, self.query_len_input,
                                        self.max_len_query, self.max_len_query)+query_emb

        passage_emb = self.self_attention(passage_emb, passage_emb, self.passage_len_input, self.passage_len_input,
                                          self.max_len_passage, self.max_len_passage)+passage_emb

        with tf.variable_scope('query_encoder'):
            query_output, query_state = bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,query_emb,sequence_length=self.query_len_input,dtype=tf.float32)

        with tf.variable_scope('passage_encoder'):
            passage_output, passage_state = bidirectional_dynamic_rnn(self.passage_cell_fw,self.passage_cell_bw,passage_emb,sequence_length=self.passage_len_input,dtype=tf.float32)

        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],sequence_length=self.alternatives_len_input[:,1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],sequence_length=self.alternatives_len_input[:,2],dtype=tf.float32)
        query_output=tf.concat(query_output,2)+tf.concat([query_emb,query_emb],-1)
        passage_output=tf.concat(passage_output,2)+tf.concat([passage_emb,passage_emb],-1)

        query_state=tf.reduce_sum(self.self_attention(query_output,query_output,self.query_len_input, self.query_len_input,
                                        self.max_len_query, self.max_len_query)+query_output,1)

        passage_state=tf.reduce_sum(self.self_attention(passage_output, passage_output, self.passage_len_input, self.passage_len_input,
                                          self.max_len_passage, self.max_len_passage)+passage_output,1)

        alternatives_state=[self.get_h(alternatives_state_0),self.get_h(alternatives_state_1),self.get_h(alternatives_state_2)]

        self.tmp=tf.concat([query_state,passage_state,alternatives_state[0]],axis=1)

        concated=tf.concat([tf.concat([query_state,passage_state,alternatives_state[0]],axis=1),
                            tf.concat([query_state, passage_state, alternatives_state[1]], axis=1),
                            tf.concat([query_state, passage_state, alternatives_state[2]], axis=1)],axis=0)

        self.middle_out=tf.transpose(tf.reshape(tf.transpose(self.output_layer_forward(concated)),[3,-1]))

    def self_attention(self,query, key, len1,len2,max_len1,max_len2):
        mask1=tf.reshape(tf.sequence_mask(len1,max_len1,dtype=tf.float32),[-1,max_len1,1])
        mask2 = tf.reshape(tf.sequence_mask(len2, max_len2, dtype=tf.float32), [-1, max_len2, 1])
        query=query * mask1
        key= key * mask2
        tmp=tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        mask2=(mask2-1)*512
        tmp=(tmp/math.pow(128,0.5))+tf.reshape(mask2,[-1,1,max_len2])
        tmp=tf.nn.softmax(tmp,-1)
        tmp=tf.matmul(tmp, key)
        tmp=tmp*mask1
        return tmp

    def compute_word_self_attention(self, passage_emb, query_state):
        tmp=tf.matmul(passage_emb,tf.reshape(query_state,[-1,self.word_dim,1]))
        mask=tf.reshape(tf.sequence_mask(self.max_len_passage,self.max_len_passage,dtype=tf.float32),[-1,self.max_len_passage,1])
        mask=512*(mask-1)
        tmp=tf.nn.softmax(tmp+mask,1)
        tmp=tmp*passage_emb
        return tmp


    def compute_word_attention(self,passage_emb,query_state):
        tmp=tf.matmul(query_state,self.attention_weight_word)
        tmp=tf.reshape(tmp,[-1,self.hidden_size*2,1])
        tmp=tf.reshape(tf.matmul(passage_emb, tmp),[-1,1,self.max_len_passage])
        mask=tf.sequence_mask(self.passage_len_input,self.max_len_passage,dtype=tf.float32)
        mask_=tf.reshape((mask-1)*(512),[-1,1,self.max_len_passage])
        tmp=tmp+mask_
        tmp = tf.nn.softmax(tmp, -1)
        self.tmp=tmp
        tmp=tf.reshape(tf.matmul(tmp,passage_emb),[-1,self.hidden_size*2])
        return tmp


    def output_layer_forward(self,input_tensor):
        cc_tensor=0
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))
            if idx==0:
                batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
                input_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, scale, 1e-3)
            if idx%2==0:
                input_tensor=input_tensor+cc_tensor
                input_tensor=tf.matmul(input_tensor,weight)+bias
                cc_tensor=input_tensor
            else:
                input_tensor = tf.matmul(input_tensor, weight) + bias
            # if idx%2==0:
            #     input_tensor=tf.nn.softsign(input_tensor)
            # else:
            if idx == self.output_layer_num - 1:
                out = input_tensor
            else:
                input_tensor = tf.nn.elu(input_tensor)
                out = tf.nn.dropout(input_tensor, keep_prob=self.keep_pro)
        return out

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def get_bi_h(self,state):
        f,b=state
        return tf.concat([self.get_h(f),self.get_h(b)],axis=1)

    def computer_loss(self):
        self.tv = tf.trainable_variables()
        for tensor in self.tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0005)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1.)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.middle_out))
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.05, 5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv), global_step=self.global_step),
                                 self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out


class CNN_model:
    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.output_layer_num = 2
        self.CNN_chanl=100
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm = config['max_grad_norm']
        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage],
                                            name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        # self.initializer = tf.random_uniform_initializer(-0.15, 0.15)
        self.initializer = tf.random_normal_initializer(stddev=0.05)
        with tf.variable_scope('word_embedding'):
            print('loading embedding')
            self.word_embedding = tf.get_variable(dtype=tf.float32,initializer=tf.constant(np.load('../../DATA/data/embedding.npy'),dtype=tf.float32), name='word_embedding',trainable=True)
            print('done')

        with tf.variable_scope('attention'):
            self.attention_weight = tf.get_variable(dtype=tf.float32,shape=[self.hidden_size * 2, self.hidden_size * 3],initializer=self.initializer, name='attention_weight')
            self.attention_weight_word=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size*2, self.hidden_size*2 ],initializer=self.initializer, name='attention_weight_word')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.CNN_chanl*5+self.word_dim, 1, self.output_layer_num, 'output_layer')
        with tf.variable_scope('pos_matrix'):
            self.pos_matrix=tf.get_variable(dtype=tf.float32, initializer=self.generate_pos_vector(100, 128).astype(np.float32), name='pos_matrix', trainable=False)
            self.pos_matrix_dy=tf.get_variable(dtype=tf.float32, shape=[100, 128], initializer=self.initializer, name='pos_matrix_dy')
        with tf.variable_scope('combine_featrue'):
            self.combine_matrix=tf.get_variable(dtype=tf.float32,initializer=self.initializer,shape=[self.CNN_chanl*5*2,self.CNN_chanl*5],name='combine_matrix')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_pos_vector(self,pos,dim):
        pos_me=np.zeros((pos,dim))
        for idx in range(pos):
            for dim_idx in range(dim):
                if dim_idx%2==0:
                    pos_me[idx,dim_idx]=math.sin(idx/math.pow(10000,dim_idx/dim))
                else:
                    pos_me[idx, dim_idx] = math.cos(idx / math.pow(10000, (dim_idx-1) / dim))
        return pos_me

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, output_size],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            elif i == 0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_betas.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_scales.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)

    def forward_step0(self):
        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)+self.pos_matrix_dy[:self.max_len_query,:]/5
        query_emb=query_emb*tf.reshape(tf.sequence_mask(self.query_len_input,self.max_len_query,dtype=tf.float32),[-1,self.max_len_query,1])

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)+self.pos_matrix_dy[:self.max_len_passage,:]/5
        passage_emb = passage_emb * tf.reshape(tf.sequence_mask(self.passage_len_input, self.max_len_passage, dtype=tf.float32),[-1, self.max_len_passage, 1])


        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])+self.pos_matrix_dy[:self.max_len_alternatives,:]/5
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])+self.pos_matrix_dy[:self.max_len_alternatives,:]/5
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])+self.pos_matrix_dy[:self.max_len_alternatives,:]/5

        # passage_emb = self.self_attention(passage_emb, passage_emb, self.passage_len_input, self.passage_len_input,self.max_len_passage, self.max_len_passage)
        # query_emb = self.self_attention(query_emb, query_emb, self.query_len_input, self.query_len_input,self.max_len_query, self.max_len_query)
        # passage_emb=self.self_attention(passage_emb,query_emb, self.passage_len_input,self.query_len_input, self.max_len_passage,self.max_len_query)

        alternatives=[self.reduce_sum_alternative(alternatives_emb_0,0),
                          self.reduce_sum_alternative(alternatives_emb_1, 1),
                          self.reduce_sum_alternative(alternatives_emb_2, 2)]

        passage=self.CNN(passage_emb,self.max_len_passage,'passage')
        query=self.CNN(query_emb,self.max_len_query,'query')
        combined=tf.matmul(tf.concat([query,passage],axis=1),self.combine_matrix)

        self.tmp=combined
        concated = tf.concat([tf.concat([combined, alternatives[0]], axis=1),
                              tf.concat([combined, alternatives[1]], axis=1),
                              tf.concat([combined, alternatives[2]], axis=1)], axis=0)

        self.middle_out = tf.transpose(tf.reshape(tf.transpose(self.output_layer_forward(concated)), [3, -1]))

    def reduce_sum_alternative(self,emb,i):
        return tf.reshape(
            tf.matmul(tf.reshape(tf.sequence_mask(
                self.alternatives_len_input[:, i], self.max_len_alternatives, dtype=tf.float32),[-1,1,self.max_len_alternatives]),emb),[-1,self.word_dim])

    def CNN(self,embedding,max_len,flag):
        embedding = tf.reshape(embedding, [-1,max_len, self.word_dim,1])
        mapping=[]
        if flag=='passage':
            for i in range(5):
                mapping.append(self.conv_and_pool(embedding,(i*3)+1,self.CNN_chanl,max_len,'passage_CNN_layer'+str(i)))
        else:
            for i in range(5):
                mapping.append(self.conv_and_pool(embedding, i + 1, self.CNN_chanl, max_len, 'query_CNN_layer'+str(i)))
        return tf.concat(mapping,1)

    def conv_and_pool(self,embedding,CONV1_HIGH,CONV1_DEEP,max_len,name):
        with tf.variable_scope(name+'_conv'):
            conv_weights = tf.get_variable('wegiht', [CONV1_HIGH, max_len, 1, CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(embedding, conv_weights, strides=[1, 1, max_len, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
        with tf.variable_scope(name+'_pool'):
            conv_shape = relu.get_shape().as_list()
            pool1 = tf.nn.max_pool(relu, ksize=[1, conv_shape[1], conv_shape[2], 1],strides=[1, conv_shape[1], conv_shape[2], 1], padding='SAME')
        return tf.reshape(pool1, [-1, self.CNN_chanl])

    def self_attention(self,query, key, len1,len2,max_len1,max_len2):
        mask1=tf.reshape(tf.sequence_mask(len1,max_len1,dtype=tf.float32),[-1,max_len1,1])
        mask2 = tf.reshape(tf.sequence_mask(len2, max_len2, dtype=tf.float32), [-1, max_len2, 1])
        tmp=tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        mask1=(mask1-1)*512
        tmp=(tmp/math.pow(128,0.5))+mask1
        tmp=tf.nn.softmax(tmp,1)
        self.tmp=tmp
        tmp=tmp*tf.reshape(mask2,[-1,1,max_len2])
        tmp=tf.reshape(tf.reduce_sum(tmp,-1),[-1,max_len1,1])
        tmp=tmp*query
        return tmp

    def compute_word_self_attention(self, passage_emb, query_state):
        tmp=tf.matmul(passage_emb,tf.reshape(query_state,[-1,self.word_dim,1]))
        mask=tf.reshape(tf.sequence_mask(self.max_len_passage,self.max_len_passage,dtype=tf.float32),[-1,self.max_len_passage,1])
        mask=512*(mask-1)
        tmp=tf.nn.softmax(tmp+mask,1)
        tmp=tmp*passage_emb
        return tmp

    def compute_word_attention(self,passage_emb,query_state):
        tmp=tf.matmul(query_state,self.attention_weight_word)
        tmp=tf.reshape(tmp,[-1,self.hidden_size*2,1])
        tmp=tf.reshape(tf.matmul(passage_emb, tmp),[-1,1,self.max_len_passage])
        mask=tf.sequence_mask(self.passage_len_input,self.max_len_passage,dtype=tf.float32)
        mask_=tf.reshape((mask-1)*(512),[-1,1,self.max_len_passage])
        tmp=tmp+mask_
        tmp = tf.nn.softmax(tmp, -1)
        tmp=tf.reshape(tf.matmul(tmp,passage_emb),[-1,self.hidden_size*2])
        return tmp


    def output_layer_forward(self,input_tensor):
        cc_tensor=0
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))

            batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
            input_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, scale, 1e-3)
            if idx%2==0:
                input_tensor=input_tensor+cc_tensor
                input_tensor=tf.matmul(input_tensor,weight)+bias
                cc_tensor=input_tensor
            else:
                input_tensor = tf.matmul(input_tensor, weight) + bias
            # if idx%2==0:
            #     input_tensor=tf.nn.softsign(input_tensor)
            # else:
            if idx == self.output_layer_num - 1:
                out = input_tensor
            else:
                input_tensor = tf.nn.elu(input_tensor)
                out = tf.nn.dropout(input_tensor, keep_prob=self.keep_pro)
        return out

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def get_bi_h(self,state):
        f,b=state
        return tf.concat([self.get_h(f),self.get_h(b)],axis=1)

    def computer_loss(self):
        self.tv = tf.trainable_variables()
        for tensor in self.tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1.)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.middle_out))
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        self.grads=grads[-1]
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.05, 5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv), global_step=self.global_step),
                                 self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out


class ATT_model:
    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.output_layer_num = 2
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm = config['max_grad_norm']
        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage],
                                            name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        # self.initializer = tf.random_uniform_initializer(-0.15, 0.15)
        self.initializer = tf.random_normal_initializer(stddev=0.25)
        with tf.variable_scope('word_embedding'):
            print('loading trained embedding')
            try:
                self.word_embedding = tf.get_variable(dtype=tf.float32,initializer=tf.constant(np.load('../../DATA/data/embedding_trained.npy'),dtype=tf.float32), name='word_embedding',trainable=True)
            except:
                print('loading trained embedding failed...')
                print('try to loading init embedding...')
                self.word_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/embedding.npy'), dtype=tf.float32), name='word_embedding',
                                                      trainable=True)
            # self.word_embedding = tf.get_variable(dtype=tf.float32,shape=[self.voc_len,self.word_dim],initializer=tf.random_normal_initializer(stddev=0.5), name='word_embedding')
            print('done')

        with tf.variable_scope('attention'):
            self.attention_weight = tf.get_variable(dtype=tf.float32,shape=[self.hidden_size * 2, self.hidden_size * 3],initializer=self.initializer, name='attention_weight')
            self.attention_weight_word=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size, self.hidden_size ],initializer=self.initializer, name='attention_weight_word')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.word_dim*2, 1, self.output_layer_num, 'output_layer')
        with tf.variable_scope('pos_matrix'):
            self.pos_matrix=tf.get_variable(dtype=tf.float32, initializer=self.generate_pos_vector(100, 128).astype(np.float32), name='pos_matrix', trainable=False)
            self.pos_matrix_dy=tf.get_variable(dtype=tf.float32, shape=[100, 128], initializer=self.initializer, name='pos_matrix_dy')
        with tf.variable_scope('combine_featrue'):
            self.combine_matrix=tf.get_variable(dtype=tf.float32,initializer=self.initializer,shape=[self.word_dim*2,self.word_dim],name='combine_matrix')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_pos_vector(self,pos,dim):
        pos_me=np.zeros((pos,dim))
        for idx in range(pos):
            for dim_idx in range(dim):
                if dim_idx%2==0:
                    pos_me[idx,dim_idx]=math.sin(idx/math.pow(10000,dim_idx/dim))
                else:
                    pos_me[idx, dim_idx] = math.cos(idx / math.pow(10000, (dim_idx-1) / dim))
        return pos_me

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, output_size],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            elif i == 0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, input_size * 4],name='auto_NN_weight_' + name + '_' + str(i),initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],name='auto_NN_bias_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_betas.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),initializer=self.initializer))
            auto_NN_scales.append(tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)

    def forward_step0(self):
        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)+self.pos_matrix_dy[:self.max_len_query,:]/10
        query_emb=query_emb*tf.reshape(tf.sequence_mask(self.query_len_input,self.max_len_query,dtype=tf.float32),[-1,self.max_len_query,1])

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)+self.pos_matrix_dy[:self.max_len_passage,:]/10
        passage_emb = passage_emb * tf.reshape(tf.sequence_mask(self.passage_len_input, self.max_len_passage, dtype=tf.float32),[-1, self.max_len_passage, 1])


        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])+self.pos_matrix_dy[:self.max_len_alternatives,:]/10
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])+self.pos_matrix_dy[:self.max_len_alternatives,:]/10
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])+self.pos_matrix_dy[:self.max_len_alternatives,:]/10

        # passage_emb = self.self_attention(passage_emb, passage_emb, self.passage_len_input, self.passage_len_input,self.max_len_passage, self.max_len_passage)
        # query_emb = self.self_attention(query_emb, query_emb, self.query_len_input, self.query_len_input,self.max_len_query, self.max_len_query)
        query=tf.reduce_sum(query_emb,1)/tf.cast(tf.reshape(self.query_len_input,[-1,1]),tf.float32)
        # passage = tf.reduce_sum(passage_emb, 1) / tf.cast(tf.reshape(self.passage_len_input, [-1, 1]), tf.float32)
        passage=self.compute_word_attention(passage_emb,query)/tf.cast(tf.reshape(self.passage_len_input,[-1,1]),tf.float32)

        alternatives=[self.reduce_sum_alternative(alternatives_emb_0,0)/tf.cast(tf.reshape(self.alternatives_len_input[:,0],[-1,1]),tf.float32),
                          self.reduce_sum_alternative(alternatives_emb_1, 1)/tf.cast(tf.reshape(self.alternatives_len_input[:,1],[-1,1]),tf.float32),
                          self.reduce_sum_alternative(alternatives_emb_2, 2)/tf.cast(tf.reshape(self.alternatives_len_input[:,2],[-1,1]),tf.float32)]

        combined=tf.matmul(tf.concat([query,passage],axis=1),self.combine_matrix)

        self.tmp=combined

        concated = tf.concat([tf.concat([combined, alternatives[0]], axis=1),
                              tf.concat([combined, alternatives[1]], axis=1),
                              tf.concat([combined, alternatives[2]], axis=1)], axis=0)

        self.middle_out = tf.transpose(tf.reshape(tf.transpose(self.output_layer_forward(concated)), [3, -1]))

    def reduce_sum_alternative(self,emb,i):
        return tf.reshape(
            tf.matmul(tf.reshape(tf.sequence_mask(
                self.alternatives_len_input[:, i], self.max_len_alternatives, dtype=tf.float32),[-1,1,self.max_len_alternatives]),emb),[-1,self.word_dim])

    def self_attention(self,query, key, len1,len2,max_len1,max_len2):
        mask1=tf.reshape(tf.sequence_mask(len1,max_len1,dtype=tf.float32),[-1,max_len1,1])
        mask2 = tf.reshape(tf.sequence_mask(len2, max_len2, dtype=tf.float32), [-1, max_len2, 1])
        tmp=tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        mask1=(mask1-1)*512
        tmp=(tmp/math.pow(128,0.5))+mask1
        tmp=tf.nn.softmax(tmp,1)
        tmp=tmp*tf.reshape(mask2,[-1,1,max_len2])
        tmp=tf.reshape(tf.reduce_sum(tmp,-1),[-1,max_len1,1])
        tmp=tmp*query
        return tmp

    def compute_word_attention(self,passage_emb,query_state):
        tmp=tf.matmul(query_state,self.attention_weight_word)
        tmp=tf.reshape(tmp,[-1,self.hidden_size,1])
        tmp=tf.reshape(tf.matmul(passage_emb, tmp),[-1,1,self.max_len_passage])
        mask=tf.sequence_mask(self.passage_len_input,self.max_len_passage,dtype=tf.float32)
        mask_=tf.reshape((mask-1)*(512),[-1,1,self.max_len_passage])
        tmp=tmp+mask_
        tmp = tf.nn.softmax(tmp, -1)
        tmp=tf.reshape(tf.matmul(tmp,passage_emb),[-1,self.hidden_size])
        return tmp


    def output_layer_forward(self,input_tensor):
        cc_tensor=0
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))

            batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
            input_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, scale, 1e-3)
            if idx%2==0:
                input_tensor=input_tensor+cc_tensor
                input_tensor=tf.matmul(input_tensor,weight)+bias
                cc_tensor=input_tensor
            else:
                input_tensor = tf.matmul(input_tensor, weight) + bias
            # if idx%2==0:
            #     input_tensor=tf.nn.softsign(input_tensor)
            # else:
            if idx == self.output_layer_num - 1:
                out = input_tensor
            else:
                input_tensor = tf.nn.elu(input_tensor)
                out = tf.nn.dropout(input_tensor, keep_prob=self.keep_pro)
        return out

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def get_bi_h(self,state):
        f,b=state
        return tf.concat([self.get_h(f),self.get_h(b)],axis=1)

    def computer_loss(self):
        self.tv = tf.trainable_variables()
        for tensor in self.tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0001)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1.)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.middle_out))
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        self.grads=grads[-1]
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.05, 5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv), global_step=self.global_step),self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out


class THE_last_attempt:

    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.output_layer_num = 5

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm = config['max_grad_norm']
        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage],
                                            name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.2, 0.2)
        # self.initializer = tf.random_normal_initializer()
        with tf.variable_scope('word_embedding'):
            print('loading trained embedding')
            try:
                self.word_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/embedding_trained.npy'), dtype=tf.float32), name='word_embedding',
                                                      trainable=True)
            except:
                print('loading trained embedding failed...')
                print('try to loading init embedding...')
                self.word_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/embedding.npy'), dtype=tf.float32), name='word_embedding',
                                                      trainable=True)
            # self.word_embedding = tf.get_variable(dtype=tf.float32,shape=[self.voc_len,self.word_dim],initializer=tf.random_normal_initializer(stddev=0.5), name='word_embedding')
            print('done')
        with tf.variable_scope('LSTM_encoder', reuse=None):
            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            try:
                print('loading trained attention weight...')
                self.attention_weight = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/att_weight.npy'), dtype=tf.float32), name='attention_weight',trainable=True)
            except:
                print('loading trained attention weight failed')
                print('init attention weight')
                self.attention_weight = tf.get_variable(dtype=tf.float32,shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer, name='attention_weight')
            print('attention weight done')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size * 5, 1, self.output_layer_num, 'output_layer')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, output_size],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))
            elif i == 0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size * 4],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, input_size * 4],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))

            auto_NN_betas.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),
                                initializer=self.initializer))
            auto_NN_scales.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),
                                initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)

    def forward_step0(self):
        # 这一部分主要是lstm编码过程
        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb = [alternatives_emb_0, alternatives_emb_1, alternatives_emb_2]

        with tf.variable_scope('query_encoder'):
            query_output, query_state = bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                        query_emb,
                                                                        sequence_length=self.query_len_input,
                                                                        dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output, passage_state = bidirectional_dynamic_rnn(self.passage_cell_fw,
                                                                            self.passage_cell_bw,
                                                                            passage_emb,
                                                                            sequence_length=self.passage_len_input,
                                                                            dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      2],
                                                                      dtype=tf.float32)

        alternatives_state = [self.get_h(alternatives_state_0),self.get_h(alternatives_state_1),self.get_h(alternatives_state_2)]
        passage_output = tf.concat(passage_output, axis=2)
        query_state = self.get_bi_h(query_state)
        passage_attentioned = self.compute_attention(passage_output,query_state)

        concated = tf.concat([tf.concat([query_state, passage_attentioned, alternatives_state[0]], axis=1),
                              tf.concat([query_state, passage_attentioned, alternatives_state[1]], axis=1),
                              tf.concat([query_state, passage_attentioned, alternatives_state[2]], axis=1)], axis=0)

        self.middle_out = tf.transpose(tf.reshape(tf.transpose(self.output_layer_forward(concated)), [3, -1]))

    def get_bi_h(self,state):
        f,b=state
        return tf.concat([self.get_h(f),self.get_h(b)],axis=1)

    def compute_attention(self, passage_outputs,query_state):
        mask=tf.sequence_mask(self.passage_len_input,self.max_len_passage,dtype=tf.float32)
        tmp=tf.matmul(query_state,self.attention_weight)
        tmp=tf.reshape(tmp,[-1,self.hidden_size*2,1])
        tmp=tf.matmul(passage_outputs,tmp)
        mask=512*(tf.reshape(mask,[-1,self.max_len_passage,1])-1)
        tmp=tf.nn.softmax(tmp+mask,1)
        self.tmp=tmp
        tmp=tf.reduce_sum(tmp*passage_outputs,1)
        return tmp

    # 小于0是训练 大于0是预测
    def output_layer_forward(self,input_tensor):
        cc_tensor=0
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))

            batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
            input_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, scale, 1e-3)
            if idx%2==0:
                input_tensor=input_tensor+cc_tensor
                input_tensor=tf.matmul(input_tensor,weight)+bias
                cc_tensor=input_tensor
            else:
                input_tensor = tf.matmul(input_tensor, weight) + bias
            if idx == self.output_layer_num - 1:
                out = input_tensor
            else:
                input_tensor = tf.nn.elu(input_tensor)
                out = tf.nn.dropout(input_tensor, keep_prob=self.keep_pro)
        return out

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def computer_loss(self):

        self.tv = tf.trainable_variables()
        for tensor in self.tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
                # if 'embedding' in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0005)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1)
        # tf.cast(tf.equal(tf.argmax(self.middle_out, axis=1), tf.reshape(self.y_input, [-1])), dtype=tf.float32) *
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.middle_out))
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        self.grads=grads[-2]
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.01, 5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv), global_step=self.global_step),
                                 self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out



class THE_lalast_attempt:
    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.output_layer_num = 5

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm = config['max_grad_norm']
        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage],
                                            name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.2, 0.2)
        # self.initializer = tf.random_normal_initializer()
        with tf.variable_scope('word_embedding'):
            print('loading trained embedding')
            try:
                self.word_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/embedding_trained.npy'), dtype=tf.float32), name='word_embedding',
                                                      trainable=True)
            except:
                print('loading trained embedding failed...')
                print('try to loading init embedding...')
                self.word_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/embedding.npy'), dtype=tf.float32), name='word_embedding',
                                                      trainable=True)
            # self.word_embedding = tf.get_variable(dtype=tf.float32,shape=[self.voc_len,self.word_dim],initializer=tf.random_normal_initializer(stddev=0.5), name='word_embedding')
            print('done')
        with tf.variable_scope('LSTM_encoder', reuse=None):
            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight_query_and_passage = tf.get_variable(dtype=tf.float32, shape=[self.word_dim, self.hidden_size * 2], initializer=self.initializer, name='attention_weight')
            self.attention_helper_weight=tf.get_variable(dtype=tf.float32,shape=[1,self.word_dim],initializer=self.initializer,name='attention_helper_weight')
            self.attention_weight_query_word=tf.get_variable(dtype=tf.float32,shape=[self.word_dim,self.word_dim],initializer=self.initializer,name='attention_weight_query_word')
            print('attention weight done')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size * 5, 1, self.output_layer_num, 'output_layer')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, output_size],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))
            elif i == 0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size * 4],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, input_size * 4],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))

            auto_NN_betas.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),
                                initializer=self.initializer))
            auto_NN_scales.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),
                                initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)

    def forward_step0(self):
        # 这一部分主要是lstm编码过程
        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb = [alternatives_emb_0, alternatives_emb_1, alternatives_emb_2]

        with tf.variable_scope('query_encoder'):
            query_output, query_state = bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                        query_emb,
                                                                        sequence_length=self.query_len_input,
                                                                        dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output, passage_state = bidirectional_dynamic_rnn(self.passage_cell_fw,
                                                                            self.passage_cell_bw,
                                                                            passage_emb,
                                                                            sequence_length=self.passage_len_input,
                                                                            dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      2],
                                                                      dtype=tf.float32)

        alternatives_state = [self.get_h(alternatives_state_0),self.get_h(alternatives_state_1),self.get_h(alternatives_state_2)]
        passage_output = tf.concat(passage_output, axis=2)
        query_state = self.get_bi_h(query_state)
        passage_attentioned = self.compute_attention(passage_output,query_emb)

        concated = tf.concat([tf.concat([query_state, passage_attentioned, alternatives_state[0]], axis=1),
                              tf.concat([query_state, passage_attentioned, alternatives_state[1]], axis=1),
                              tf.concat([query_state, passage_attentioned, alternatives_state[2]], axis=1)], axis=0)

        self.middle_out = tf.transpose(tf.reshape(tf.transpose(self.output_layer_forward(concated)), [3, -1]))

    def get_bi_h(self,state):
        f,b=state
        return tf.concat([self.get_h(f),self.get_h(b)],axis=1)

    def compute_attention(self, passage_outputs, query_emb):
        mask_passage=tf.sequence_mask(self.passage_len_input,self.max_len_passage,dtype=tf.float32)
        query_emb=tf.reshape(query_emb,[-1,self.word_dim])
        tmp=tf.matmul(query_emb, self.attention_weight_query_and_passage)
        tmp=tf.transpose(tf.reshape(tmp,[-1,self.max_len_query,self.hidden_size*2]),[0,2,1])

        tmp=tf.matmul(passage_outputs,tmp)
        mask_passage=512*(tf.reshape(mask_passage,[-1,self.max_len_passage,1])-1)
        tmp=tf.nn.softmax(tmp+mask_passage,1)
        tmp*=self.compute_query_attention(query_emb)
        tmp=tf.reshape(tf.reduce_sum(tmp,-1),[-1,self.max_len_passage,1])
        self.tmp = tmp
        tmp=tf.reduce_sum(tmp*passage_outputs,1)
        return tmp

    def compute_query_attention(self,query_emb):
        tmp=tf.matmul(self.attention_helper_weight,self.attention_weight_query_word)
        tmp=tf.matmul(query_emb,tf.transpose(tmp))
        tmp=tf.reshape(tmp,[-1,1,self.max_len_query])
        mask=tf.reshape(tf.sequence_mask(self.query_len_input,self.max_len_query,dtype=tf.float32),[-1,1,self.max_len_query])
        mask=(mask-1)*512
        tmp+=mask
        tmp=tf.nn.softmax(tmp,-1)
        return tmp



    # 小于0是训练 大于0是预测
    def output_layer_forward(self,input_tensor):
        cc_tensor=0
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))

            batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
            input_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, scale, 1e-3)
            if idx%2==0:
                input_tensor=input_tensor+cc_tensor
                input_tensor=tf.matmul(input_tensor,weight)+bias
                cc_tensor=input_tensor
            else:
                input_tensor = tf.matmul(input_tensor, weight) + bias
            if idx == self.output_layer_num - 1:
                out = input_tensor
            else:
                input_tensor = tf.nn.elu(input_tensor)
                out = tf.nn.dropout(input_tensor, keep_prob=self.keep_pro)
        return out

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def computer_loss(self):

        self.tv = tf.trainable_variables()
        for tensor in self.tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
                # if 'embedding' in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.0005)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1)
        # tf.cast(tf.equal(tf.argmax(self.middle_out, axis=1), tf.reshape(self.y_input, [-1])), dtype=tf.float32) *
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.middle_out))
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        self.grads=grads[-2]
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.01, 5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv), global_step=self.global_step),
                                 self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out

class THE_lalalast_attempt:
    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.word_dim = config['word_dim']
        self.voc_len = config['voc_len']

        self.max_len_passage = config['max_len_passage']
        self.max_len_query = config['max_len_query']
        self.max_len_alternatives = config['max_len_alternatives']
        self.output_layer_num = 5

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 600, 0.96, staircase=True)

        self.max_grad_norm = config['max_grad_norm']
        self.flag = tf.constant(dtype=tf.int32, value=0, name='flag')
        self.auto_NN = {}
        self.build_placeholder()
        self.build_para()
        self.forward_step0()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')

        self.query_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_query], name='query_input')
        self.passage_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len_passage],
                                            name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')
        self.alternatives_input = tf.placeholder(dtype=tf.int32, shape=[None, 3, self.max_len_alternatives],
                                                 name='alternatives_input')
        self.alternatives_len_input = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='alternatives_len_input')

        self.whether_train = tf.placeholder(dtype=tf.int32, name='whether_train')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.2, 0.2)
        # self.initializer = tf.random_normal_initializer()
        with tf.variable_scope('word_embedding'):
            print('loading trained embedding')
            try:
                self.word_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/embedding_trained.npy'), dtype=tf.float32), name='word_embedding',
                                                      trainable=True)
            except:
                print('loading trained embedding failed...')
                print('try to loading init embedding...')
                self.word_embedding = tf.get_variable(dtype=tf.float32, initializer=tf.constant(
                    np.load('../../DATA/data/embedding.npy'), dtype=tf.float32), name='word_embedding',
                                                      trainable=True)
            # self.word_embedding = tf.get_variable(dtype=tf.float32,shape=[self.voc_len,self.word_dim],initializer=tf.random_normal_initializer(stddev=0.5), name='word_embedding')
            print('done')
        with tf.variable_scope('LSTM_encoder', reuse=None):
            self.query_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.query_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.passage_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)
            self.passage_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_pro,
                output_keep_prob=self.keep_pro, state_keep_prob=self.keep_pro)

            self.alternatives_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

        with tf.variable_scope('attention'):
            self.attention_weight_query_state_and_passage_emb = tf.get_variable(dtype=tf.float32, shape=[self.hidden_size * 2,self.word_dim], initializer=self.initializer, name='attention_weight')
            print('attention weight done')
        with tf.variable_scope('output_layer'):
            self.generate_NN(self.hidden_size * 5, 1, self.output_layer_num, 'output_layer')

        self.ema = tf.train.ExponentialMovingAverage(0.9, self.global_step)
        self.maintain_average_op = self.ema.apply(tf.trainable_variables())

    def generate_NN(self, input_size, output_size, layers, name):
        auto_NN_weights = []
        auto_NN_biases = []
        auto_NN_betas = []
        auto_NN_scales = []
        for i in range(layers):
            if i == layers - 1:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, output_size],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[output_size],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))
            elif i == 0:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size, input_size * 4],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))
            else:
                auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4, input_size * 4],
                                                       name='auto_NN_weight_' + name + '_' + str(i),
                                                       initializer=self.initializer))
                auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * 4],
                                                      name='auto_NN_bias_' + name + '_' + str(i),
                                                      initializer=self.initializer))

            auto_NN_betas.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_beta_' + name + '_' + str(i),
                                initializer=self.initializer))
            auto_NN_scales.append(
                tf.get_variable(dtype=tf.float32, shape=[1], name='auto_NN_scale_' + name + '_' + str(i),
                                initializer=self.initializer))
        self.auto_NN[name] = zip(auto_NN_weights, auto_NN_biases, auto_NN_betas, auto_NN_scales)

    def forward_step0(self):
        # 这一部分主要是lstm编码过程
        query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query_input)

        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)

        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 0, :])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])

        alternatives_emb = [alternatives_emb_0, alternatives_emb_1, alternatives_emb_2]

        with tf.variable_scope('query_encoder'):
            query_output, query_state = bidirectional_dynamic_rnn(self.query_cell_fw, self.query_cell_bw,
                                                                        query_emb,
                                                                        sequence_length=self.query_len_input,
                                                                        dtype=tf.float32)

        query_state = self.get_bi_h(query_state)
        passage_emb=self.compute_emb_attetion(passage_emb,query_state)

        with tf.variable_scope('passage_encoder'):
            passage_output, passage_state = bidirectional_dynamic_rnn(self.passage_cell_fw,
                                                                            self.passage_cell_bw,
                                                                            passage_emb,
                                                                            sequence_length=self.passage_len_input,
                                                                            dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb[0],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb[1],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      1], dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb[2],
                                                                      sequence_length=self.alternatives_len_input[:,
                                                                                      2],
                                                                      dtype=tf.float32)

        alternatives_state = [self.get_h(alternatives_state_0),self.get_h(alternatives_state_1),self.get_h(alternatives_state_2)]

        passage=self.get_bi_h(passage_state)

        concated = tf.concat([tf.concat([query_state, passage, alternatives_state[0]], axis=1),
                              tf.concat([query_state, passage, alternatives_state[1]], axis=1),
                              tf.concat([query_state, passage, alternatives_state[2]], axis=1)], axis=0)

        self.middle_out = tf.transpose(tf.reshape(tf.transpose(self.output_layer_forward(concated)), [3, -1]))

    def get_bi_h(self,state):
        f,b=state
        return tf.concat([self.get_h(f),self.get_h(b)],axis=1)

    def compute_emb_attetion(self,passage_emb,query_state):
        tmp=tf.reshape(tf.matmul(query_state,self.attention_weight_query_state_and_passage_emb),[-1,self.word_dim,1])
        tmp=tf.sigmoid(tf.matmul(passage_emb,tmp))*2
        self.tmp=tmp
        tmp*=passage_emb

        return tmp


    # 小于0是训练 大于0是预测
    def output_layer_forward(self,input_tensor):
        cc_tensor=0
        for idx, (weight, bias, beta, scale) in enumerate(self.auto_NN['output_layer']):
            # weight, bias, beta, scale=tf.cond(self.whether_train<self.flag,
            #                                   lambda : self.doto_the_ctrl(weight,bias,beta,scale,True),
            #                                   lambda :self.doto_the_ctrl(weight,bias,beta,scale,False))

            batch_mean, batch_var = tf.nn.moments(input_tensor, [0])
            input_tensor = tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, beta, scale, 1e-3)
            if idx%2==0:
                input_tensor=input_tensor+cc_tensor
                input_tensor=tf.matmul(input_tensor,weight)+bias
                cc_tensor=input_tensor
            else:
                input_tensor = tf.matmul(input_tensor, weight) + bias
            if idx == self.output_layer_num - 1:
                out = input_tensor
            else:
                input_tensor = tf.nn.elu(input_tensor)
                out = tf.nn.dropout(input_tensor, keep_prob=self.keep_pro)
        return out

    def get_h(self, state):
        c, h = state
        h = tf.reshape(h, [-1, self.hidden_size])
        return h

    def computer_loss(self):

        self.tv = tf.trainable_variables()
        for tensor in self.tv:
            try:
                if not 'bias' in tensor.name and not 'embedding' in tensor.name and not 'beta' in tensor.name and 'attention' not in tensor.name:
                # if 'embedding' in tensor.name:
                    print(tensor)
                    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(tensor))
            except:
                pass
        self.loss_l2=tf.add_n(tf.get_collection("losses"))
        # self.loss_l2 = tf.constant(1)
        # tf.cast(tf.equal(tf.argmax(self.middle_out, axis=1), tf.reshape(self.y_input, [-1])), dtype=tf.float32) *
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.middle_out))
        tf.add_to_collection("losses", loss)
        self.loss = tf.add_n(tf.get_collection("losses"))

    def _train(self):
        grads = tf.gradients(self.loss, self.tv)
        self.grads=grads[-2]
        # grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=tf.clip_by_value(self.lr, 0.01, 5))
        self.train_op = tf.group(opt.apply_gradients(zip(grads, self.tv), global_step=self.global_step),
                                 self.maintain_average_op)

    def val_(self, sess, tensor_list, feed_dict, flag):
        max_len = len(feed_dict['query'])
        scale = 10000
        out = []
        for i in range(10000):
            out.append(sess.run(tensor_list, feed_dict={
                self.query_input: feed_dict['query'][i * scale:min((i + 1) * scale, max_len)],
                self.query_len_input: feed_dict['query_len'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_input: feed_dict['passage'][i * scale:min((i + 1) * scale, max_len)],
                self.passage_len_input: feed_dict['passage_len'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_input: feed_dict['alternative'][i * scale:min((i + 1) * scale, max_len)],
                self.alternatives_len_input: feed_dict['alternative_len'][i * scale:min((i + 1) * scale, max_len)],
                self.y_input: feed_dict['answer'][i * scale:min((i + 1) * scale, max_len)],
                self.keep_pro: 1,
                self.whether_train: -1}))
            if (i + 1) * scale >= max_len:
                break
        return out