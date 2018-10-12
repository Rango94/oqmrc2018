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

class Oqmrc_Model:

    def __init__(self,config):
        self.hidden_size=config['hidden_size']
        self.num_layers=config['num_layers']
        self.word_dim=config['word_dim']
        self.voc_len=config['voc_len']
        self.max_len=config['max_len']

        self.global_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(config['lr'], self.global_step, 800, 0.96, staircase=True)

        self.max_grad_norm=config['max_grad_norm']
        self.build_placeholder()
        self.build_para()
        self.forward()
        self.computer_loss()
        self._train()

    def build_placeholder(self):
        self.query_input=tf.placeholder(dtype=tf.int32,shape=[None,None],name='query_input')
        self.passage_input=tf.placeholder(dtype=tf.int32,shape=[None,None],name='passage_input')

        self.query_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='query_len_input')
        self.passage_len_input = tf.placeholder(dtype=tf.int32, shape=[None], name='passage_len_input')

        self.y_input=tf.placeholder(dtype=tf.int32,shape=[None],name='y_input')
        self.alternatives_input=tf.placeholder(dtype=tf.int32,shape=[None,3,None],name='alternatives_input')
        self.alternatives_len_input=tf.placeholder(dtype=tf.int32,shape=[None,3],name='alternatives_len_input')

    def build_para(self):
        self.initializer = tf.random_uniform_initializer(-0.25, 0.25)
        with tf.variable_scope('word_embedding'):
            self.word_embedding=tf.get_variable(dtype=tf.float32,shape=[self.voc_len,self.word_dim],initializer=self.initializer,name='word_embedding')
        with tf.variable_scope('LSTM_encoder',reuse=None):
            self.query_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size) for _ in range(self.num_layers)])
            self.passage_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size) for _ in range(self.num_layers)])
            self.alternatives_cell=tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size) for _ in range(self.num_layers)])
        with tf.variable_scope('attention'):
            self.attention_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size,self.hidden_size],initializer=self.initializer,name='attention_weight')
        with tf.variable_scope('feature_extract'):
            self.generate_NN(self.hidden_size*2,self.hidden_size,3)
        with tf.variable_scope('similar'):
            self.similar_weight=tf.get_variable(dtype=tf.float32,shape=[self.hidden_size,self.hidden_size*2],name='similar_weight')

    def generate_NN(self,input_size,output_size,layers,times=2):
        self.auto_NN_weights=[]
        self.auto_NN_biases=[]
        for i in range(layers):
            if i==layers-1:
                self.auto_NN_weights.append(tf.get_variable(dtype=tf.float32,shape=[input_size*times,output_size],name='auto_NN_weight_'+str(i),initializer=self.initializer))
                self.auto_NN_biases.append(tf.get_variable(dtype=tf.float32,shape=[output_size],name='auto_NN_bias_'+str(i),initializer=self.initializer))
            else:
                self.auto_NN_weights.append(tf.get_variable(dtype=tf.float32, shape=[input_size * times, input_size * times],name='auto_NN_weight_' + str(i),initializer=self.initializer))
                self.auto_NN_biases.append(tf.get_variable(dtype=tf.float32, shape=[input_size * times],name='auto_NN_bias_' + str(i),initializer=self.initializer))

    def forward(self):
        query_emb=tf.nn.embedding_lookup(self.word_embedding,self.query_input)
        passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage_input)
        alternatives_emb_0 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:,0,:])
        alternatives_emb_1 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 1, :])
        alternatives_emb_2 = tf.nn.embedding_lookup(self.word_embedding, self.alternatives_input[:, 2, :])
        with tf.variable_scope('query_encoder'):
            query_output, query_state = dynamic_rnn(self.query_cell, query_emb,sequence_length=self.query_len_input,dtype=tf.float32)
        with tf.variable_scope('passage_encoder'):
            passage_output,passage_state=dynamic_rnn(self.passage_cell, passage_emb,sequence_length=self.passage_len_input,dtype=tf.float32)
        with tf.variable_scope('alternatives_encoder'):
            alternatives_output_0, alternatives_state_0 = dynamic_rnn(self.alternatives_cell, alternatives_emb_0,sequence_length=self.alternatives_len_input[:,0], dtype=tf.float32)
            alternatives_output_1, alternatives_state_1 = dynamic_rnn(self.alternatives_cell, alternatives_emb_1,sequence_length=self.alternatives_len_input[:,1],dtype=tf.float32)
            alternatives_output_2, alternatives_state_2 = dynamic_rnn(self.alternatives_cell, alternatives_emb_2,sequence_length=self.alternatives_len_input[:,2],dtype=tf.float32)
        query_state=self.get_h(query_state)

        alternatives_state=tf.concat([self.get_h(alternatives_state_0),self.get_h(alternatives_state_1),self.get_h(alternatives_state_2)],axis=1)

        passage_state=self.compute_attention(query_state, passage_output, self.passage_len_input)

        self.final_state=tf.concat([query_state,passage_state],axis=1)

        tmp=tf.matmul(tf.reshape(alternatives_state,shape=[-1,self.hidden_size]),self.similar_weight)
        tmp=tf.reshape(tmp,shape=[-1,self.hidden_size*2,3])
        self.middle_out=tf.matmul(tf.reshape(self.final_state,shape=[-1,1,self.hidden_size*2]),tmp)

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
        self.loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(self.y_input,shape=[-1,1]),logits=self.middle_out)

    def _train(self):
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        self.train_op = opt.apply_gradients(zip(grads, trainable_variables),global_step=self.global_step)


