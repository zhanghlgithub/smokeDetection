# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:15:27 2018

@author: zhl
"""

import cv2
import numpy as np
import testModel

VIDEO_PATH0 = "G:/DeepLearningCode/DataSet/medias/videos/nosmoke3.avi"
VIDEO_PATH1 = "G:/DeepLearningCode/DataSet/mediasOfSomke/videos/self1.mp4"
VIDEO_PATH2 = "G:/photo/video_006.mp4"
OUTPUT_WINDOW = "OUTPUT_WINDOW"
MOG = "MOG2"
FINALLY_WINDOW = "FINALLY_WINDOW"
DEBUG_SHOW = True
DEBUG_PRINT = True
BLOCK_WIDTH = 32    #切割小块的宽度
BLOCK_HEIGHT = 24   #切割小块的高度
AVERAGE_S_THRESHOLD = 70 #色彩S的均值的阈值
HSV_V_BLOCK_COUNT = 50
FRAME_SKIP = 1
FRAME_SIZE = (32,24)

def readVideo(PATH):
    cap = cv2.VideoCapture()
    cap.open(PATH)
    if cap.isOpened() is None:
        print("could not read video")
        return None
    else:
        return cap

def getVideoSize(capture):
    
    ret,frame = capture.read()
    if frame is None:
        return None
    else:
        return frame.shape[:2]
    
def calc_direction(list1, list2):
    
    
    if (len(list1) < 23 or (len(list2) < 23)):
        if DEBUG_PRINT:
            print("return 7.......1")
        return 7
    if (len(list1[0]) < 23 or (len(list2[0]) < 23)):
        if DEBUG_PRINT:
            print("return 7.......2")
        return 7
    s = 0.0
    for w in range(BLOCK_WIDTH):
        for h in range(BLOCK_HEIGHT):
            s += list1[h][w] - list2[h][w]
    s = s / (w * h)
    return s

def get_move_toward(list_frames, m, n):
    
    bias = 2
#    if m<bias or n<bias or m> FRAME_SIZE[0]-BLOCK_WIDTH-bias or n>FRAME_SIZE[1]-BLOCK_HEIGHT-bias:
#        return 7;
    
    #代表当碎片块
    block = list_frames[1][n:(n+BLOCK_HEIGHT), m:(m+BLOCK_WIDTH)] 
    #当前区域正向上
    block1 = list_frames[0][(n-bias):(n+BLOCK_HEIGHT-bias), m:(m+BLOCK_WIDTH)]
    #当前区域右向上
    block2 = list_frames[0][(n-bias):(n+BLOCK_HEIGHT-bias), (m+bias):(m+BLOCK_WIDTH+bias)]
    #当前区域右上方
    block3 = list_frames[0][(n-bias):(n+BLOCK_HEIGHT-bias), (m-bias):(m+BLOCK_WIDTH-bias)]
    #当前区域右下方
    block4 = list_frames[0][(n+bias):(n+BLOCK_HEIGHT+bias), (m+bias):(m+BLOCK_WIDTH+bias)]
    #当前区域正下方
    block5 = list_frames[0][(n+bias):(n+BLOCK_HEIGHT+bias), m:(m+BLOCK_WIDTH)]
    #当前区域左下方
    block6 = list_frames[0][(n+bias):(n+BLOCK_HEIGHT+bias), (m-bias):(m+BLOCK_WIDTH-bias)]
    #当前区域正左方
    block7 = list_frames[0][(n):(n+BLOCK_HEIGHT), (m-bias):(m+BLOCK_WIDTH-bias)]        
    #当前区域正右方
    block8 = list_frames[0][n:(n+BLOCK_HEIGHT), (m+bias):(m+BLOCK_WIDTH+bias)]
    
    list_result = []    
    r1 = calc_direction(block, block1)
    list_result.append(r1)
    r2 = calc_direction(block, block2)
    list_result.append(r2)
    r3 = calc_direction(block, block3)
    list_result.append(r3)
    r4 = calc_direction(block, block4)
    list_result.append(r4)
    r5 = calc_direction(block, block5)
    list_result.append(r5)
    r6 = calc_direction(block, block6)
    list_result.append(r6)
    r7 = calc_direction(block, block7)
    list_result.append(r7)
    r8 = calc_direction(block, block8)
    list_result.append(r8)
    if DEBUG_PRINT:
        print(list_result)
    index = list_result.index(min(list_result))
    return index

def handleFrame(capture):
    
    #创建MOG模型
    MOG2 = cv2.createBackgroundSubtractorMOG2()
    
    #创建显示窗口    
    cv2.namedWindow(OUTPUT_WINDOW,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(MOG,cv2.WINDOW_AUTOSIZE)
    
    #创建对图像进行形态学操作的卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    #获取视频的长/宽
    if getVideoSize(capture):
        height,width = getVideoSize(capture)
        if DEBUG_PRINT:
            print(height,width)
            
    HSV_V_all_block = []
    two_gray_frames = []
    frame_count = 0    
    while True:
        ret,frame = capture.read()
        if DEBUG_SHOW:
            cv2.imshow(OUTPUT_WINDOW,frame)
            
        if 0xFF == 27 or frame is None:
            print("ESC")
            break
        smooth_kernel = np.ones((5, 5), np.float32)/25
        smooth_frame = cv2.filter2D(frame, -1, smooth_kernel)
               
        gray_frame = cv2.cvtColor(smooth_frame,cv2.COLOR_BGR2GRAY) #将图像转换成灰度图像
        if len(two_gray_frames) > FRAME_SKIP:   #逻辑：始终保持two_gray_frames中是当前帧和前一帧
            two_gray_frames.pop(0)
        two_gray_frames.append(gray_frame)
        
        hsv_frame = cv2.cvtColor(smooth_frame, cv2.COLOR_BGR2HSV_FULL) #将图像转换成HSV图像
        
        bsmaskMOG2 = MOG2.apply(gray_frame)
        #对图像进行开操作,开操作的过程就是先腐蚀后膨胀的过程，                
        bsmaskMOG2 = cv2.morphologyEx(bsmaskMOG2,cv2.MORPH_OPEN,kernel)
        
        #对图像进行二值化
        ret,bsmaskMOG2 = cv2.threshold(bsmaskMOG2,0,255,cv2.THRESH_BINARY)
        if DEBUG_SHOW:
            cv2.imshow(MOG,bsmaskMOG2)            
            
        HSV_V_each_block = [] #存储每一个小块的V色彩通道的平均值
        HSV_V_50_block = np.array(0) #
        for block_width in range(0,width,BLOCK_WIDTH):
            for block_height in range(0,height,BLOCK_HEIGHT):
                clip_threshold = bsmaskMOG2[block_height:(block_height+BLOCK_HEIGHT),
                                            block_width:(block_width+BLOCK_WIDTH)]
                clip_hsv = hsv_frame[block_height:(block_height+BLOCK_HEIGHT),
                                            block_width:(block_height+BLOCK_WIDTH)]
                
                
                HSV_V_each_block.append(np.average(clip_hsv[:,:,2]))
                
                #逻辑：有白色区域的小块则认为是运动区域
                if clip_threshold.any():
                    
                    average_S = np.average(clip_hsv[:, :, 1])
                    average_V = np.average(clip_hsv[:, :, 2])
                    
                    #逻辑：如果该区域的S的均值小于70，则该区域有可能是烟雾区域
                    if average_S < AVERAGE_S_THRESHOLD:
                        HSV_V_all_block_ndarray = np.array(HSV_V_all_block)
                        if frame_count > HSV_V_BLOCK_COUNT - 1:
                            HSV_V_50_block = HSV_V_all_block_ndarray[:,block_width//20]
                        elif frame_count > 0:
                            HSV_V_50_block = HSV_V_all_block_ndarray[:frame_count,
                                                                     block_width//20]
                    
                    if np.average(HSV_V_50_block) - average_V < 0:
                        
                        candidate_block = frame[block_height:(block_height+BLOCK_HEIGHT),
                                            block_width:(block_width+BLOCK_WIDTH)]
                        
                        if frame_count > FRAME_SKIP:
                            
                            toward_up_num = get_move_toward(two_gray_frames,block_width,block_height)
                            if DEBUG_PRINT:
                                print("frame_count:%d" % frame_count)
                                print("tower_up_num:%d" % toward_up_num)
                                
                            if 0 <= toward_up_num < 3:
                                
                                candidate_block_feed = np.reshape(candidate_block,(1,-1))
                                reshape = np.reshape(candidate_block_feed,[1,32,24,3])
                                
#                                print(reshape)
                                #把candidate_block_feed喂入神经网络模型
                                result = testModel.check_image(reshape)
                                if DEBUG_PRINT:
                                    print(result[0][0])
                                    print(result[0][1])
                                if result[0][0] > result[0][1]:
                                    if DEBUG_PRINT:
                                        print("显示此时的长度和宽度：")
                                        print(block_height,block_width)
                                        cv2.rectangle(frame, (block_width, block_height), 
                                                      (block_width+BLOCK_WIDTH, block_height+BLOCK_HEIGHT), 
                                                      (25, 150, 155),3)
                                        
                                        cv2.putText(frame,"Smoke",(block_width, block_height),
                                                    1,1.0,(25,100,200),2)
                                        if DEBUG_SHOW:
                                            cv2.imshow(FINALLY_WINDOW,frame)
                                        
        if frame_count > HSV_V_BLOCK_COUNT - 1:
            HSV_V_all_block.pop(0)
            HSV_V_all_block.append(HSV_V_each_block)
        else:
            HSV_V_all_block.append(HSV_V_each_block)
                            
        frame_count += 1
        cv2.waitKey(5)
        
    
    cv2.destroyAllWindows()
        
        
def main():
    cap = readVideo(VIDEO_PATH1)
    if cap:
        handleFrame(cap)
        cap.release()

if __name__ == "__main__":
    main()
