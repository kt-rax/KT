# -*- coding: utf-8 -*-
### this make model be ready to be trained and evluated on you custom dataset
#### Putting everything together
#### Let's now write the main function which performs the training and the validation

from detection.engine import train_one_epoch,evaluate
import torch
import utils
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T
import torchvision
from torchviz import make_dot
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from kt_package.Personal_module import print_time
#from torchvision import transforms

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
        #    img, target = self.transforms(img, target)
            img = self.transforms(img)

        #    img, target = self.transforms(img, target)  TypeError: __call__() takes 2 positional arguments but 3 were given
        #https://stackoverflow.com/questions/62341052/typeerror-call-takes-2-positional-arguments-but-3-were-given-to-train-ra
        return img, target

    def __len__(self):
        return len(self.imgs)    

def get_transform(train):    
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.ToPILImage())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.ToTensor())
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
    

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torch.load(r'./Experiment2/Experiment2_ maskrcnn_resnet50_fpn_model_010.pkl')
    model.load_state_dict(torch.load(r'./Experiment2/Experiment2_ maskrcnn_resnet50_fpn_model_param_010.pkl'))

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

'''
#def main():
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('PennFudanPed',get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed',get_transform(train=False))

# split the dataset in train and test set
indices  = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

'''

 # train on the GPU or on the CPU, if a GPU is not available
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device = torch.device('cuda')
# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)

# let's train it for 10 epochs
num_epochs = 5

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    for Experiment in range(4,5):
        run()
        print_time()
        
        coco_evaluator_history = []
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            coco_evaluator_ = evaluate(model, data_loader_test, device=device)
            coco_evaluator_history = np.append(coco_evaluator_history,coco_evaluator_)

        # model save
        os.mkdir(os.path.join('%s/Experiment%d/'%(os.getcwd(),Experiment+1)))
        
        model_save_name = './Experiment%d/Experiment%d_maskrcnn_resnet50_fpn_model_%03d.pkl'%(Experiment+1,Experiment+1,epoch+1)
        torch.save(model,model_save_name)
        
        model_param_save_name = './Experiment%d/Experiment%d_maskrcnn_resnet50_fpn_model_param_%03d.pkl'%(Experiment+1,Experiment+1,epoch+1)
        torch.save(model.state_dict(),model_param_save_name)
        
        #make_dot(pred[0]['boxes'],params=dict(model.named_parameters()),show_attrs=True,show_saved=True).render('faster_resnet50_fpn',format='png')
  
         
    print("That's it!")
    print_time()



'''check datasets
import cv2
img = cv2.imread('PennFudanPed/PedMasks/FudanPed00047_mask.png')
img = (img+1)**7


col,row,chan = img.shape

for i in range(0,col): 
    for j in range(0,row):
        for k in range(0,chan):
            if img[i,j,k] > 250:
                img[i,j,k] = 240
     
#cv2.imshow('FudanPed00001_mask',img)
cv2.imwrite('test.png',img)
#cv2.waitKey(1)
#cv2.destroyAllWindows()
'''






