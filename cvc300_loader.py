import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import os
import glob
import random

class CVC300_loader(Dataset):
    def __init__(self,iw=512,ih=512,augmentation=False,phase='test'):
        super().__init__()
        self.iw = iw
        self.ih = ih
        self.__get_path__(phase)
        self.augmentation = augmentation
        self.phase = phase
        self.num_class = 2
        self.img_normalization = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.transform_AUG =  A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
        ])

        self.mask_normalization = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.img)
    
    def __get_path__(self,phase,split=0.8):
        fname_list = os.listdir('./dataset/CVC-300/images/')
        self.img = ['./dataset/CVC-300/images/' + x for x in fname_list]
        self.mask = ['./dataset/CVC-300/masks/' + x for x in fname_list]     
    
    def __getitem__(self, index):
        img = cv2.imread(self.img[index])
        img = cv2.resize(img,(self.iw,self.ih))
        label = cv2.imread(self.mask[index],cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label,(self.iw,self.ih),interpolation=cv2.INTER_NEAREST)  
        
        if self.augmentation:
            augmentations = self.transform_AUG(image=img,mask=label)
            img = augmentations['image']
            label = augmentations["mask"]
        
        #init_mask
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray,(5,5),0)
        ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        init_mask = (th>0.5)*1
        
        init_mask = self.mask_normalization(init_mask)
        init_mask = torch.cat([1-init_mask,init_mask],dim=0)
        img = self.img_normalization(img)
        label = self.mask_normalization(label)        
        
        return self.img[index], img, label, init_mask