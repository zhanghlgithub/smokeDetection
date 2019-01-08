# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 08:49:23 2018

@author: zhl
"""
import tensorflow as tf
import numpy as np
import forward 
import makeData 
import debug
import os
import testModel

Debug = debug.DebugTool(False)

BATCH_SIZE = 40
STEPS = 1000
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "smoke_model"

def backward():
    '''function:反向传播优化网络'''
    Debug.print_error()
    X = tf.placeholder(tf.float32,shape=[BATCH_SIZE,
                                         forward.IMAGE_WIDTH,
                                         forward.IMAGE_HEIGTH,
                                         forward.NUM_CHANNELS])
    Debug.print_error()
    y_real = tf.placeholder(tf.float32,
                            shape=[None,forward.OUTPUT_NODE])       #X对应的真实标签
    y_predict = forward.forward(X, True)                            #X对应的预测值
    print("前向传播网络搭建完毕")
   
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(y_real,1),logits=y_predict)

    with tf.name_scope("train_op"):    
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        
    saver = tf.train.Saver()
    Debug.print_error()
    
    #获取数据集tfrecords_name
    image_data = makeData.ImageData(makeData.tfrecords_name, BATCH_SIZE)
    image_data.read_tfRecord()
    img_batch,label_batch = image_data.get_data()
    
    #创建会话sess
    Debug.print_error()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
       
        for index in range(STEPS):
            x_data, y_data = sess.run([img_batch,label_batch])
            
#            Debug.print_current_value(y_data)
            reshape_xs = np.reshape(x_data,[BATCH_SIZE,
                                            forward.IMAGE_WIDTH,
                                            forward.IMAGE_HEIGTH,
                                            forward.NUM_CHANNELS])
            Debug.print_error()
            _, loss_value = sess.run([train_op, loss], 
                                           feed_dict={X:reshape_xs, y_real:y_data})
            
            #每训练100轮保存当前的模型
            if (index + 1) % 20 == 0:
                print("After %d training steps,loss on training batch is %g." % (index, loss_value))
                #把当前会话保存到指定的路径下面
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                           global_step=index)
                
                testModel.model_test()
                
        
        coord.request_stop()
        coord.join(threads)
        
                
            
if __name__ == "__main__":
    backward()
    
    