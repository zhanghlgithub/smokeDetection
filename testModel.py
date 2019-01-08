# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:06:38 2018

@author: zhl
"""
import tensorflow as tf
import numpy as np
import backward
import forward
import makeData
import debug

Debug = debug.DebugTool(False)
BATCH_SIZE = 100    
STEP = 10

def model_test():
    '''function:测试模型的准确度
    ''' 
    with tf.Graph().as_default() as g:
        Debug.print_error()
        x = tf.placeholder(tf.float32,[BATCH_SIZE,
                                       forward.IMAGE_WIDTH,
                                       forward.IMAGE_HEIGTH,
                                       forward.NUM_CHANNELS])
    
        y = forward.forward(x,False)
        y_ = tf.placeholder(tf.float32,[None,2])
        Debug.print_error()
        saver = tf.train.Saver()       
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 
        
        #获取数据集
        image_data = makeData.ImageData(makeData.test_tfrecords_name, BATCH_SIZE)
        image_data.read_tfRecord()
        img_batch,label_batch = image_data.get_data()
        
        Debug.print_error()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
           
            Debug.print_error()
            
            if ckpt and ckpt.model_checkpoint_path:
                Debug.print_error()
                saver.restore(sess,ckpt.model_checkpoint_path)
                
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for index in range(STEP):
                    
                    
                    x_data, y_data = sess.run([img_batch,label_batch])
                    
                    reshape = np.reshape(x_data,[BATCH_SIZE,forward.IMAGE_WIDTH,
                                                   forward.IMAGE_HEIGTH,forward.NUM_CHANNELS])
                   
                    accuracy_score = sess.run(accuracy,feed_dict={x:reshape,y_:y_data})
                    print(accuracy_score)
                    
                coord.request_stop()
                coord.join(threads)
                
            else:
                print('No checkpoint file found')
               
    

        