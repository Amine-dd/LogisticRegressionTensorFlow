# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:40:13 2019

@author: Bouslimi
"""
import tensorflow.compat.v1 as tf
n_steps = 1000
 #Generating synthetic data
import numpy as np
# Generate synthetic data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2,))
# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N//2,))
y_ones = np.ones((N//2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Train model
    for i in range(n_steps):
        feed_dict = {x:x_np,y:y_np}
        _,summary,loss = sess.run([train_op,merged,l],feed_dict = feed_dict)
        print('loss %f ' % loss)
        train_writer.add_summary(summary,i)
