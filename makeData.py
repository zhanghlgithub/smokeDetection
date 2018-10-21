# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 08:50:09 2018

Function:make data

@author: zhl
"""
import cv2
#import tensorflow as tf
import numpy as np
import os 
#import random

train_smoke_path = "G:/DeepLearningCode/DataSet/mediasOfSomke/pictures/smoke_train_32x24/"
train_nosmoke_path = "G:/DeepLearningCode/DataSet/mediasOfSomke/pictures/nosmoke_train_32x24/"
test_smoke_path = "G:/DeepLearningCode/DataSet/mediasOfSomke/pictures/smoke_test_32x24/"
test_none_path = "G:/DeepLearningCode/DataSet/mediasOfSomke/pictures/nosmoke_test_32x24/"

def load_images(path):
    '''
    load images from directiry
    return a list of images data
    '''
#    img_list = []
#    dirpath, dirnames, filenames = os.walk(path)
    img_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        
        for filename in filenames:
            img = cv2.imread(path + filename)
            #使用5*5的核对图像进行高斯滤波
            img2 = cv2.GaussianBlur(img,(5,5),0)
            img_flat = np.reshape(img2,(1,-1))
            img_list.append(img_flat)
    
    return img_list

#功能：制作数据集
def make_data():
    
    #把图片信息加载出来
    train_smoke_images = load_images(train_smoke_path)
    train_none_smoke_images = load_images(train_nosmoke_path)
    test_smoke_images = load_images(test_smoke_path)
    test_none_smoke_images = load_images(test_none_path)

    total_train_images_list = []
    total_train_labels_list = []
    total_test_images_list = []
    total_test_labels_list = []
    #给每个数据集添加标签，采用的是热码的方式，有烟label：[1,0];
    #无烟lable:[0,1]
    for i in range(len(train_smoke_images)):
        total_train_images_list.extend(
            np.array(
                train_smoke_images[i],
                dtype = np.float32                 
            )
        )
        total_train_labels_list.append([1, 0])
    
    for i in range(len(train_none_smoke_images)):
        total_train_images_list.extend(
            np.array(
                train_none_smoke_images[i],
                dtype=np.float32
            )
        )
        total_train_labels_list.append([0,1])
        
    for i in range(len(test_smoke_images)):
        total_test_images_list.extend(test_smoke_images[i]) 
        total_test_labels_list.append([1,0])
        
    for i in range(len(test_none_smoke_images)):
        total_test_images_list.extend(test_none_smoke_images[i])
        total_test_labels_list.append([0,1])
    
    #将图像数据由列表格式转换成ndarray格式
    _total_train_images = np.array(total_train_images_list, dtype=np.float32) / 255
    _total_train_labels = np.array(total_train_labels_list, dtype=np.float32)
    _total_test_images = np.array(total_test_images_list, dtype=np.float32) / 255
    _total_test_labels = np.array(total_test_labels_list, dtype=np.float32)
        
    return _total_train_images,_total_train_labels,\
           _total_test_images,_total_test_labels 

#if __name__ == "__main__":
#    total_train_images, total_train_labels, \
#        total_test_images, total_test_labels = make_data()
#    print(total_train_images)
    

