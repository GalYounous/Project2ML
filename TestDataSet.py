from torch.utils.data import Dataset

from PIL import Image

import os
import glob
import torchvision.transforms as T
import torch
import numpy as np
import random as random
mean=torch.tensor([84.91024075/255, 84.17390113/255, 75.42844544/255])
std=torch.tensor([49.44622809/255, 47.72316675/255, 47.68987178/255])

# normalize image [0-1] (or 0-255) to zero-mean unit standard deviation
normalize = T.Compose([T.ToTensor(),T.Normalize(mean, std)])
# we invert normalization for plotting later
std_inv = 1 / (std + 1e-7)
unnormalize = T.Normalize(-mean * std_inv, std_inv)

        
class TestDataSet(Dataset):

    def __init__(self, dataset_root='test_set_images',samples=range(50)):
        # prepare data
        self.data = []                                  # list of (image path)

        for i in samples:
            self.data.append( dataset_root+"/test_{}/test_{}.png".format(i+1,i+1)  )
            
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, x):
        img_path= self.data[x] #Get paths
        img = Image.open(img_path)

        #Normalization
        img = normalize(img)
        
        return img
