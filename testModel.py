# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:06:38 2018

@author: zhl
"""
import tensorflow as tf
import numpy as np
import backward as bk
import forward as fw
import makeData as md
import time
import cv2

DEBUG_PRINT = True

'''
def compute_accuracy():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[1000,32,24,3])
        y_ = tf.placeholder(tf.float32,[1000,2])
        y = fw.forward(x,False,None)
        
        ema = tf.train.ExponentialMovingAverage(bk.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        #获取训练数据集
        total_train_images, total_train_labels, \
        total_test_images, total_test_labels = md.make_data()
        print(len(total_test_images))
        reshape = np.reshape(total_test_images,[1000,32,24,3])
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(bk.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    #accuracy_score = sess.run(accuracy,feed_dict={x:reshape,y_:total_test_labels})
                    accuracy_score = sess.run(y,feed_dict={x:reshape,y_:total_test_labels})
                    #print("After %s training step(s),test accuracy = %g" % (global_step,accuracy_score))
                    print(accuracy_score)
                    print(accuracy_score[0][0])
                    print(accuracy_score[0][1])
                else:
                    print('No checkpoint file found')
                    return 
            time.sleep(5)

'''   
total_train_images, total_train_labels, \
        total_test_images, total_test_labels = md.make_data()
         
def check_image():
     
    with tf.Graph().as_default() as g:
        print(len(total_test_images))
        
        x = tf.placeholder(tf.float32,[1000,32,24,3])
        y = fw.forward(x,False,None)
        y_ = tf.placeholder(tf.float32,[1000,2])
        
        ema = tf.train.ExponentialMovingAverage(bk.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)       
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(bk.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                reshape = np.reshape(total_test_images,[1000,32,24,3])
                saver.restore(sess,ckpt.model_checkpoint_path)
                accuracy_score = sess.run(accuracy,feed_dict={x:reshape,y_:total_test_labels})
                if DEBUG_PRINT:
                    print(accuracy_score)
#                    print(accuracy_score[0][0])
#                    print(accuracy_score[0][1])
                return accuracy_score
            else:
                print('No checkpoint file found')
                return 
            

def main():
#    src = cv2.imread("G:/DeepLearningCode/DataSet/mediasOfSomke/pictures/smoke_test_32x24/3_2012-07-17_15-15-44-clip12-0.jpg")

    check_image()
    
if __name__ == "__main__":
    main()
    

        