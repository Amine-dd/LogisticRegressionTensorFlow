# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:00:25 2019

@author: Bouslimi 
Linear Regression with tensorflow
"""
import tensorflow.compat.v1 as tf
N = 100
tf.disable_eager_execution()
with tf.name_scope('placeholders'):
    x = tf.placeholder(tf.float32, (N,2))
    y = tf.placeholder(tf.float32,(N,))
    
with tf.name_scope('weights'):
    W = tf.Variable(tf.random_normal((2,1)))
    b = tf.Variable(tf.random_normal((1,)))
    
with tf.name_scope('prediction'):
    y_logit = tf.squeeze(tf.matmul(x,W)+b)
    y_one_prob = tf.sigmoid(y_logit)
    y_pred = tf.round(y_one_prob)
with tf.name_scope('loss'):
    #compute the cross enropy for every term
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit,labels=y)
    l = tf.reduce_sum(entropy)
    
with tf.name_scope('optim'):
    train_op = tf.train.AdamOptimizer(.01).minimize(l)
    
with tf.name_scope('summaries'):
    tf.summary.scalar('loss',l)
    merged = tf.summary.merge_all()
    
    train_writer = tf.summary.FileWriter('/tmp/logistic-train',tf.get_default_graph())