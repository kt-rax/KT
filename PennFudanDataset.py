#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:40:43 2021

@author: root
"""
import os
import numpy as np
import torch
from PIL import Image
import torch

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms):
        self.root = root
        self.transforms = transforms
        # load all images files,sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root,'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root,'PedMasks'))))        
        
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root,'PNGImages',self.imgs[idx])
        mask_path = os.path.join(self.root,'PedMasks',self.masks[idx])
        img = Image .open(img_path).convert('RGB')
        # note that we haven't converted the mask to RGB
        # beacuse each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy
        mask = np.array(mask)
        # instances are ended as different colors
        obj_ids = np.unique(mask)
        # first id is the background,so remove it
        obj_idx = obj_ids[1:]
        
        # split the color-encoded mask into a set
        masks = mask = obj_ids[:,None,None]
        
        # get bounding box  coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(mask[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin,xmax,ymin,ymax])
            
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs),dtype=torch.int64)
        masks = torch.as_tensor(masks,dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        # suppose all instances are not crowd
        iscrowd = torch.zero_((num_objs,),dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms is not None:
            img,target = self.transforms(img,target)
            
        return img,target
    
    def __len__(self):
        return len(self.imgs)

