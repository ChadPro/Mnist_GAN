# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import cv2

""" DATASET
"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

""" Param
"""
DATASET_LEN = len(x_train)
STDDEV = 0.01
STDDEV1 = 0.1
BATCH_SIZE = 64
IMAGE_SIZE = 784
init_v1 = tf.truncated_normal_initializer(stddev=STDDEV)
# init_v1 = tf.contrib.layers.xavier_initializer()

""" 生成器
"""
Z = tf.placeholder(tf.float32, shape=[None,IMAGE_SIZE], name="z")
with tf.variable_scope("gen"):
    G_w1 = tf.get_variable(name="G_w1", shape=[IMAGE_SIZE,128], initializer=init_v1)
    G_b1 = tf.get_variable(name="G_b1", shape=[128], initializer=tf.constant_initializer(0.))
    G_w2 = tf.get_variable(name="G_w2", shape=[128,784], initializer=init_v1)
    G_b2 = tf.get_variable(name="G_b2", shape=[784], initializer=tf.constant_initializer(0.))
theta_G = [G_w1, G_w2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_w1) + G_b1)
    G_logits = tf.matmul(G_h1, G_w2) + G_b2
    outputs = tf.tanh(G_logits) #-1~1
    return G_logits, outputs

""" 判别器
"""
X = tf.placeholder(tf.float32, shape=[None,784], name="x")
with tf.variable_scope("dis"):
    D_w1 = tf.get_variable(name="D_w1", shape=[784,128], initializer=init_v1)
    D_b1 = tf.get_variable(name="D_b1", shape=[128], initializer=tf.constant_initializer(0.))
    D_w2 = tf.get_variable(name="D_w2", shape=[128,1], initializer=init_v1)
    D_b2 = tf.get_variable(name="D_b2", shape=[1], initializer=tf.constant_initializer(0.))
theta_D = [D_w1, D_w2, D_b1, D_b2]

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_w1) + D_b1)
    D_logits = tf.matmul(D_h1, D_w2) + D_b2
    outputs = tf.nn.sigmoid(D_logits) #0~1
    return outputs, D_logits

""" 算法
"""
# 生成器生成数据
G_logit, G_output = generator(Z)
# 识别器识别来自真实及虚拟的数据
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_output)
# 损失函数
D_loss = - tf.reduce_mean(tf.log(D_real)+tf.log(1. - D_fake))
G_loss = - tf.reduce_mean(tf.log(D_fake))
tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

merged = tf.summary.merge_all()
logwriter = tf.summary.FileWriter("./log/", tf.get_default_graph())
gan_saver = tf.train.Saver()

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    c = 0
    for it in range(100000000):
        if (c+1)*BATCH_SIZE < DATASET_LEN:
            train_data = x_train[c*BATCH_SIZE : (c+1)*BATCH_SIZE]
            train_data = np.reshape(train_data, [BATCH_SIZE, -1])
            train_data = train_data / 255. * 2 - 1

            _, __, D_loss_curr, G_loss_curr, summary_str = sess.run([D_solver, G_solver, D_loss, G_loss, merged], feed_dict={X:train_data, Z:sample_Z(BATCH_SIZE,IMAGE_SIZE)})

            if it%1000 == 0:
                logwriter.add_summary(summary_str, it)
                print "####### " + str(it) + "#######"
                print "D_loss : " + str(D_loss_curr)
                print "G_loss : " + str(G_loss_curr)

            if it%3000 == 0:
                gan_saver.save(sess, "./models/model.ckpt")

            c += 1
        else:
            c=0