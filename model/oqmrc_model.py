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
import numpy as np
import random as rd

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
