from torch.utils.data import Dataset

from PIL import Image

import os
import glob
import torchvision.transforms as T
import torch
import numpy as np
import random as random

def_mean=torch.tensor([84.91024075/255, 84.17390113/255, 75.42844544/255])
def_std=torch.tensor([49.44622809/255, 47.72316675/255, 47.68987178/255])


        
class CustomDataSet(Dataset):

    def __init__(self, dataset_root='training', im_trans=T.Compose([]),gt_trans=T.Compose([]),samples=range(100),mean=def_mean, std=def_std):

         #Set up normalization
        self.normalize=T.Normalize(mean,std)

        self.image_transforms = T.Compose([T.ToTensor(),im_trans,self.normalize]) # to tensor, transform and normalize
        self.gt_transforms = T.Compose([T.ToTensor(),gt_trans])# to tensor, some transforms only
        # prepare data
        self.data = []                                  # list of tuples of (image path, ground truth path)
        path_im = dataset_root+"/images/satImage_"
        path_gt = dataset_root+"/groundtruth/satImage_"
        for i in samples:
            self.data.append( (path_im+"{:03d}.png".format(i+1) , path_gt+"{:03d}.png".format(i+1)) )
            
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, x):
        img_path, gt_path = self.data[x] #Get paths
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        gt_array = np.array(gt)
        threshold = 128 #Threshold 
        gt = (gt_array > threshold).astype(np.uint8)*255 #Remove anti aliasing

        #Transforms
        seed = np.random.randint(2147483647) #In order to have the same rotations for image and ground truth
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.image_transforms(img)
        
        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transforms(gt)
        
        return img, gt
