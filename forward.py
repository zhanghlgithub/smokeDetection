# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:53:45 2018

@author: zhl
"""

#import cv2
import tensorflow as tf
#import numpy as np

FC_SIZE = 1024
OUTPUT_NODE = 2

def weight_variable(shape,regularizer):
    #随机生成权重
    initial = tf.truncated_normal(shape,stddev=0.01)
    #正则化操作
    if regularizer != None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(initial))
    return tf.Variable(initial)

def bias_variable(shape):
    #生成偏重的值为0.1
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    #卷积操作，x:要训练的数据；w:卷积核的大小
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    #最大池化操作
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#前向传播过程，返回预测值prediction
def forward(x, train, regularizer):
    
    #参数x：喂入神经网络的一张图片
    
    #第一层卷积
    w_conv1 = weight_variable([5, 5, 3, 32],regularizer)
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1) #输出图片是32*24，卷积层的深度：32
    h_pool1 = max_pool_2x2(h_conv1) #最大池化后图片的大小是16*12，卷积层的深度不变，仍为32
    
    #第二层卷积
    w_conv2 = weight_variable([5, 5, 32, 64],regularizer)
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2) #输出图片是16*12，卷积层的深度：64
    h_pool2 = max_pool_2x2(h_conv2) #最大池化后图片的大小是8*6，卷积层的深度不变，仍为64
    
    #第三层卷积
    w_conv3 = weight_variable([5, 5, 64, 128],regularizer)
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3) #输出图片是8*6，卷积层的深度：128
    h_pool3 = max_pool_2x2(h_conv3) #最大池化后图片的大小是4*3，卷积层的深度不变，仍为128
    
    #图片的二维数据拉直
    h_pool3_shape = h_pool3.get_shape().as_list()
    node = h_pool3_shape[1] * h_pool3_shape[2] * h_pool3_shape[3]
    reshape = tf.reshape(h_pool3,[h_pool3_shape[0],node])
    
    #全连接层:
    #1、由输入层到第一层网络，第一层网络的神经元的个数：FC_SIZE
    w_fc1 = weight_variable([node,FC_SIZE],regularizer)
    b_fc1 = bias_variable([FC_SIZE])
    h_fc1 = tf.nn.relu(tf.matmul(reshape,w_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, 0.5)
    
    #2、由第一层神经元到输出层：输出层神经元的个数：OUTPUT_NODE
    w_fc2 = weight_variable([FC_SIZE,OUTPUT_NODE],regularizer)
    b_fc2 = bias_variable([OUTPUT_NODE])
    #softmax激活函数shiprediction的值位于0~1之间 
    prediction = tf.matmul(h_fc1, w_fc2) + b_fc2 
    
    return prediction
