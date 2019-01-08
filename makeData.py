# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 08:50:09 2018

Function:make data

@author: zhl
"""
import cv2
import tensorflow as tf
import numpy as np
import os 
import random
import debug

Debug = debug.DebugTool(False)

#train
pre_path = "./dataSet/train"
image_path = "./Image/train"
tfrecords_name = "./tfrecords/tf_train.tfrecords"
txt_path = "./Image_txt/train.txt"

#test
test_pre_path = "./dataSet/test"
test_image_path = "./Image/test"
test_tfrecords_name = "./tfrecords/tf_test.tfrecords"
test_txt_path = "./Image_txt/test.txt"    

class MakeData():
    ''' function：制作训练和测试的数据集 '''
    def __init__(self, pre_path, image_path, tfRecordName, image_size):
        '''
        Arg:
            - pre_path: 预处理图像路径
            - image_path: 源图像路径
            - tfRecordName:tfRecord文件保存路径
            - image_size: 图像的尺寸,
        '''
        self.path = pre_path
        self.image_path = image_path
        self.tfRecordName = tfRecordName
        self.image_size = image_size 
        self.count = 0
        self.labels = 0
        
        
    def makeTxt(self, txt_path):
        '''function:把图像名和对应标签对应写入txt文件'''
        self.data_txt = txt_path        
        txt_list = []
        for dirname in os.listdir(self.path):
            self.labels += 1
            Debug.print_current_value(self.labels)
            Debug.print_current_value(dirname)
            for filename in os.listdir(self.path + "/" + dirname):
                merge_data = filename + " " + str(self.labels) + "\n"
#                Debug.print_current_value(merge_data)
                txt_list.append(merge_data)
        
        #随机打乱txt_list的值
        random.shuffle(txt_list)
        fp = open(self.data_txt,'w')
        fp.truncate()   #清空文件内容
        fp.close()
        fp = open(self.data_txt,'a')
        for i in range(len(txt_list)):
            fp.write(txt_list[i])
        fp.close()
    
    def write_tfRecord(self):
        '''function: 把数据和对应标签制作成tfRecord文件'''
        writer = tf.python_io.TFRecordWriter(self.tfRecordName)
        
        fp = open(self.data_txt, "r")
        contents = fp.readlines()
        fp.close()
        for content in contents:
#            Debug.print_current_value(content)
            value = content.split()
            image_path = os.path.join(self.image_path, value[0])
#            Debug.print_current_value(image_path)
            src = cv2.imread(image_path)
#            src = cv2.resize(src,self.image_size, interpolation=cv2.INTER_LINEAR)           #将读出的图片转换成指定大小
#            src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)      #图像转换成灰度图像
#            src = cv2.equalizeHist(src)                     #图像的均值化操作2018.11.4号 
#            Debug.print_current_value(src)
            img_raw = src.tobytes()
            Debug.print_current_value(value[1])
            labels = [0] * 2
            labels[int(value[1]) - 1] = 1
            Debug.print_current_value(labels)
            example = tf.train.Example(features=tf.train.Features(feature={
                            'img_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=labels))                    
                        }))
            writer.write(example.SerializeToString())
                
        writer.close()
        print("write tfrecord successful")

#make_data = MakeData(pre_path, image_path, tfrecords_name, (32, 24))
#make_data.makeTxt(txt_path)  
#make_data.write_tfRecord()   
        
#make_data = MakeData(test_pre_path, test_image_path, test_tfrecords_name, (32, 24))
#make_data.makeTxt(test_txt_path)  
#make_data.write_tfRecord()   
            
class ImageData():
            
    def __init__(self, tfRecordName,batch_szie):
        self.tfRecordName = tfRecordName
        self.batch_size = batch_szie
        
    def read_tfRecord(self):
        '''function: 读取tfRecord文件'''
        filename_queue = tf.train.string_input_producer([self.tfRecordName])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                                   'label' : tf.FixedLenFeature([2], tf.int64),
                                                   'img_raw' : tf.FixedLenFeature([], tf.string)
                                                   })
        
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img.set_shape([32 * 24 * 3])          
        img = tf.cast(img, tf.float32) 
        self.image = img / 127.5 - 1            #图像的数值位于[-1,1]
        self.labels = tf.cast(features['label'],tf.float32)
    
    def get_data(self):
        image_batch,label_batch = tf.train.shuffle_batch([self.image,self.labels],
                                             batch_size=self.batch_size,
                                             num_threads=2,
                                             capacity = 100,
                                             min_after_dequeue= 70 )
        return image_batch, label_batch

def rename(path):
    '''function:对文件进行重新命名，path：当前的目录'''
    i = 0
    dir_file = os.listdir(path)
    for file_name in dir_file:
        old_name = path + "/" + file_name
        i += 1
        new_name = path + "/test_smoke_" + str(i) + ".jpg" 
        
        os.rename(old_name,new_name)
#path = "./dataSet/test/smoke_test_32x24"
#rename(path)

