#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:57:05 2021

@author: root
"""

# in reference/detection,we have a number of helper functions to simplify training and evaluating detection models.
# here,we will use references/detection/engine.py,references/detection/utils.py and references/detectiion/transforms.py.
# just copy everything under references/detection to your folder and use them here
from torchvision import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


### Testing forward() method (optional)
## before iterating over the dataset,it's good to see what the model expects during training and inference time on sample data
import torch
from torch import utils
from torch.utils.data import dataloader
import torchvision
import PennFudanDataset

model= torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

dataset = PennFudanDataset.PennFudanDataset('PennFudanPed',get_transform(train=True))
#data_loader = torch.utils.data.dataloader(dataset,batch_size=2,shuffle=True,num_workers =4,collate_fn= utils.collate_fn)
data_loader = torch.utils.data.dataloader(dataset,batch_size=2,shuffle=True,num_workers =4)