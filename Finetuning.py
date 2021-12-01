#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:49:28 2021

@author: root
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#### 1.finetuning from a pretrained model

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has number_classes which is user-defined
num_classes = 2 # 1 class(person) + background
# get number of input features for the classiifer
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)