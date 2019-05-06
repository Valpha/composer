import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 简单卷积层，为方便本章教程叙述，固定部分参数
def conv_layer(input,
               channels_in,  # 输入通道数
               channels_out):  # 输出通道数

    weights = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[channels_out]))
    conv = tf.nn.conv2d(input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    act = tf.nn.relu(conv + biases)
    return act


# 简化全连接层
def fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
    act = tf.matmul(input, weights) + biases

    if use_relu:
        act = tf.nn.relu(act)
    return act


# max pooling 层
def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
