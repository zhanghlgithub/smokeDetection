# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 08:49:23 2018

@author: zhl
"""
import tensorflow as tf
import numpy as np
import forward as fw 
import makeData as md
import random
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005 #学习率
LEARNING_RATE_DECAY = 0.99 #学习衰减率
REGULARIZER = 0.0001
STEPS = 2001 
MOVING_AVERAGE_DECAY = 0.99 #滑动平均值的衰减率
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "smoke_model"

def backward():
    
    xs = tf.placeholder(tf.float32,[BATCH_SIZE,32, 24, 3]) #喂给神经网络的数据集xs
    ys = tf.placeholder(tf.float32,[None,2])  #xs对应的标签label
    prediction = fw.forward(xs,True,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)
        
    #定义损失函数loss【疑问：不明白损失函数为什么这样定义？？？】
#    cross_entropy = tf.reduce_mean(
#        -tf.reduce_sum(ys * tf.log(prediction), 1)
#        )
    #求loss函数并对其进行正则化操作
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=tf.argmax(ys,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    #使用梯度下降法优化损失函数loss，用指数衰减的方式更改学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            240,
            LEARNING_RATE_DECAY,
            staircase=True
            )
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    #给权重w设定影子值
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name="train")
        
    saver = tf.train.Saver()
   
    #创建会话sess
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #获取数据集
        total_train_images, total_train_labels, \
        total_test_images, total_test_labels = md.make_data()
#        total_train_images, total_train_labels = make_data() 
        #round_num = 100
        for index in range(2001):
            batch_xs = []
            batch_ys = []
            for i in range(BATCH_SIZE):
                #在总的训练数据集上随机抽取其中的一个数据
                rand_num = random.randint(0, total_train_images.shape[0] - 1)
                batch_xs.append(total_train_images[rand_num])
                batch_ys.append(total_train_labels[rand_num])
            
            reshape_xs = np.reshape(batch_xs,[BATCH_SIZE,32,24,3])
            
            _, loss_value, step = sess.run([train_op, loss, global_step], 
                                           feed_dict={xs:reshape_xs, ys:batch_ys})
            #每训练100轮保存当前的模型
            if index % BATCH_SIZE == 0:
                print("After %d training steps,loss on training batch is %g." % (index, loss_value))
                #把当前会话保存到指定的路径下面
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                           global_step=index)
                
            
if __name__ == "__main__":
    
    backward()
#    total_train_images = []
#    total_train_images = make_data()
#    batch_xs = []
#    for i in range(5):
#        print(len(total_train_images[i]))
#        batch_xs.append(total_train_images[i])
#    
#    reshape = np.reshape(batch_xs,[-1,32,24,3])
#    print(reshape)  
#    forward(reshape,True)