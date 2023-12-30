import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import os
import glob
import random

class ISIC_loader(Dataset):
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
        phase = ['train','val','test']
        for ph in phase:
            list_img = os.listdir('./dataset/ISIC_2018/'+ph+'_image/')
            list_img_split = [item.split('.') for item in list_img]
            list_mask = [item[0]+'_segmentation.png' for item in list_img_split]
            img_path = './dataset/ISIC_2018/'+ph+'_image'
            mask_path = './dataset/ISIC_2018/'+ph+'_mask'
            all_img_path = [os.path.join(img_path,x) for x in list_img]
            all_mask_path = [os.path.join(mask_path,x) for x in list_mask]
            self.__write_txt(fname='./dataset/ISIC_2018/'+ph+'_img.txt',items=all_img_path)
            self.__write_txt(fname='./dataset/ISIC_2018/'+ph+'_mask.txt',items=all_mask_path)
    
    def __read_txt(self,path):
        with open(path) as f:
            list = [line.strip() for line in f.readlines()]
        return list
    
    def __get_path__(self,phase,split=0.8):
        if not os.path.exists('./dataset/ISIC_2018/'+phase+'_img.txt'):
            self.__gen_train_test(split=split)
        if phase=='train':
            self.img = self.__read_txt('./dataset/ISIC_2018/train_img.txt') 
            self.mask = self.__read_txt('./dataset/ISIC_2018/train_mask.txt')
        elif phase=='val':
            self.img = self.__read_txt('./dataset/ISIC_2018/val_img.txt') 
            self.mask = self.__read_txt('./dataset/ISIC_2018/val_mask.txt')   
        else:
            self.img = self.__read_txt('./dataset/ISIC_2018/test_img.txt') 
            self.mask = self.__read_txt('./dataset/ISIC_2018/test_mask.txt')    
    
    def __getitem__(self, index):
        img = cv2.imread(self.img[index])
        #print(self.img[index])
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
        #init = np.expand_dims(init,axis=2)
        #init_mask = np.concatenate([~init,init],axis=2)
        #print(init_mask.shape)
        init_mask = self.mask_normalization(init_mask)
        init_mask = torch.cat([1-init_mask,init_mask],dim=0)
        img = self.img_normalization(img)
        label = self.mask_normalization(label)        
        
        return self.img[index], img, label, init_mask