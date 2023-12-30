import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import os
import glob
import random

class BUS_loader(Dataset):
    def __init__(self,iw=512,ih=512,augmentation=False,phase='train'):
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
    
    def __write_txt(self,fname,items):
        with open(fname, 'w') as fp:
            for item in items:
                # write each item on a new line
                fp.write("%s\n" % item)

    def __gen_train_test(self,split=0.8):
        cat_list = ['normal','benign','malignant']
        train_img = []
        train_mask = []
        test_img = []
        test_mask = []
        for i in range(len(cat_list)):
            list_img = os.listdir('./dataset/BUSI_clean/'+cat_list[i]+'/image/')
            list_img_split = [item.split('.') for item in list_img]
            list_mask = [item[0]+'_mask.png' for item in list_img_split]
            img_path = './dataset/BUSI_clean/'+cat_list[i]+'/image'
            mask_path = './dataset/BUSI_clean/'+cat_list[i]+'/mask'
            all_img_path = [os.path.join(img_path,x) for x in list_img]
            all_mask_path = [os.path.join(mask_path,x) for x in list_mask]
            if i>0:
                indices = [x for x in range(len(all_img_path))]
                random.shuffle(indices)
                train_img.extend(all_img_path[:int((len(indices))*split)])
                train_mask.extend(all_mask_path[:int((len(indices))*split)])
                test_img.extend(all_img_path[int((len(indices))*split):])
                test_mask.extend(all_mask_path[int((len(indices))*split):])
            else:
                train_img.extend(all_img_path)
                train_mask.extend(all_mask_path)
        self.__write_txt(fname='./dataset/BUSI_clean/train_img.txt',items=train_img)
        self.__write_txt(fname='./dataset/BUSI_clean/train_mask.txt',items=train_mask)
        self.__write_txt(fname='./dataset/BUSI_clean/test_img.txt',items=test_img)
        self.__write_txt(fname='./dataset/BUSI_clean/test_mask.txt',items=test_mask)
    
    def __read_txt(self,path):
        with open(path) as f:
            list = [line.strip() for line in f.readlines()]
        return list
    
    def __get_path__(self,phase,split=0.8):
        if not os.path.exists('./dataset/BUSI_clean/'+phase+'.txt'):
            self.__gen_train_test(split=split)
        if phase=='train':
            self.img = self.__read_txt('./dataset/BUSI_clean/train_img.txt') 
            self.mask = self.__read_txt('./dataset/BUSI_clean/train_mask.txt')          
        else:
            self.img = self.__read_txt('./dataset/BUSI_clean/test_img.txt') 
            self.mask = self.__read_txt('./dataset/BUSI_clean/test_mask.txt')     
    
    def __getitem__(self, index):
        img = cv2.imread(self.img[index])
        img = cv2.resize(img,(self.iw,self.ih))
        label = cv2.imread(self.mask[index],cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label,(self.iw,self.ih),interpolation=cv2.INTER_NEAREST)  
        
        if self.augmentation:
            augmentations = self.transform_AUG(image=img,mask=label)
            img = augmentations['image']
            label = augmentations["mask"]
        
        img = self.img_normalization(img)
        label = self.mask_normalization(label)        
        
        return self.img[index], img, label