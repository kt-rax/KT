#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:50:59 2021

@author: root
"""

#### 2. Modifying the model to add a different backbone

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a  pre-trained model for classification and return 
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN need to know the number of output channels in backbone. For mobilenet_v2,it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5X3 anchors per spatial
# location, with 5 different size and 3 different aspect ratios. We have a Tuple[Tuple[int]]
# because each feature map could potentially have different sizes and aspect ratios
anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))

# let's define what are the feature maps that we will use to perform the region of interest cropping , as wwell
# as the size of the crop after rescaling. if your backbone returns a Tensor,featmap_names is expected to be [0].
# More generally, the backbone should return an OrdereDict[Tensor],and the featmap_names you can choose which feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
# put the pieces together inside a FasterRCNN model
model =FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler)
