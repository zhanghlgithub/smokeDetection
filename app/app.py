# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:47:03 2019

@author: zhl
"""

import tensorflow as tf
import numpy as np
import forward
import debug
import cv2

Debug = debug.DebugTool(False)
BATCH_SIZE = 10     
    

def app(reshape_image):
    '''function:使用模型预测单张图片
    arg:
        - reshape_image:要预测的图片-->[1, 32, 24, 3]
    ''' 
    with tf.Graph().as_default() as g:
       
        x = tf.placeholder(tf.float32,[1, 32, 24, 3])    
        y_predict = forward.forward(x,False)
        
        saver = tf.train.Saver()             
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('../model')
           
            if ckpt and ckpt.model_checkpoint_path:
                
                reshape = np.reshape(reshape_image,[1, 32, 24, 3])
                saver.restore(sess,ckpt.model_checkpoint_path)
                result = sess.run(y_predict, feed_dict={x:reshape})
                
                Debug.print_current_value(result) #打印调试
                return result
            else:
                print('No checkpoint file found')
                return None

#def main():
#    image = "./app_iamge/1.jpg"
#    image = cv2.imread(image)
#    image = cv2.resize(image, (32, 24))
#        
#    app(image)
#    
#if __name__ == "__main__":
#    main()
