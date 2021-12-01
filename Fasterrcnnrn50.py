# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import torchvision
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import  random
import os
from torchvision import models
import logging
from torchviz import make_dot

# 加载模型 
device = torch.device("cpu") 
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device) 

# 设置成评估模式

model.eval()   
   

# 定义 Pytorch 官方给的类别名称，有些是 'N/A' 是已经去掉的类别
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

#  获取单张图片的预测结果
def get_prediction(img_path, threshold):
    img = Image.open(img_path)      # Load the image  加载图片
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img)            # Apply the transform to the image  转换成 torch 形式
    pred = model([img.to(device)])  # Pass the image to the model  开始推理
 

    #make_dot(model)
    make_dot(pred[0]['boxes'],params=dict(model.named_parameters()),show_attrs=True,show_saved=True).render('faster_resnet50_fpn',format='png')
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]       # Get the Prediction Score  获取预测的类别
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes  获取各个类别的边框
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())                                         #  获取各个类别的分数

    # Get list of index with score greater than threshold.
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  #  判断分数大于阈值对于的分数的最大索引
    #  因为预测后的分数是从大到小排序的，只要找到大于阈值最后一个的索引值即可
    pred_boxes = pred_boxes[:pred_t+1]  
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


#  根据预测的结果绘制边框及类别
def object_detection_api(img_path, threshold=0.5, rect_th=1, text_size=0.5, text_th=1):
    boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
    
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    result_dict = {}  #  用来保存每个类别的名称及数量
    for i in range(len(boxes)):
        color = tuple(random.randint(0, 255) for i in range(3))
        cv2.rectangle(img,
                      (int(boxes[i][0][0]),int(boxes[i][0][1])),  #boxes[i][0],fixed  error: (-5:Bad argument) - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
                      (int(boxes[i][1][0]),int(boxes[i][1][1])),  #boxes[i][1],   
                      color=color,
                      thickness=rect_th)  # Draw Rectangle with the coordinates

        cv2.putText(img,
                    pred_cls[i],
                    (int(boxes[i][0][0]),int(boxes[i][0][1])),    #boxes[i][0], fixedd - Can't parse 'org'. Sequence item with index 0 has a wrong type
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    text_size,  
                    color,
                    thickness=text_th)  # Write the prediction class

        #  将各个预测的结果保存到一个字典里
        if pred_cls[i] not in result_dict:
            result_dict[pred_cls[i]] = 1
        else:
            result_dict[pred_cls[i]] += 1
        print(result_dict)
    plt.figure(figsize=(6.4, 4.8))  # display the output image
    #(float, float), default: rcParams["figure.figsize"] (default: [6.4, 4.8]) Width, height in inches.
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('result.jpg')
    plt.show()
    

if __name__ == "__main__":
    
    # 单词检测 
    object_detection_api('./people.jpg', threshold=0.5)
    
    # 多次运行测试时间 
    FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'    
    #运行5次测试模型的检测时间 
    logging.basicConfig(format=FORMAT)
    d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
    for i in range(1,5):
        logger = logging.getLogger('位置')
        logger.warning('' ,extra=d)
        object_detection_api('./people.jpg', threshold=0.5)