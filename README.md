# 功能：检测视频中的烟雾

## 一、制作数据集
### 1、dataSet文件中存放未经处理的原始的训练数据和测试数据，尺寸大小32*24，用于生成txt文件，把dataSet文件下的train和test中的子文件合并就生成Image文件下train和test文件，分别存放训练数据集和测试数据集，和Image_txt文件通过调用makeData.py文件中类函数共同生成tfrecords文件存放到tfrecords文件下

## 二、训练
### 1、forward.py：前向运算网络（VGG）,forward.py：反向优化并把训练好的模型保存到Model文件下
## 三、测试与模型应用
### 1、app文件下app.py测试单张图片，handleVideo.py文件处理视频，并将训练好的模型运用到视频分析中；对烟雾的识别准确率达到98%。