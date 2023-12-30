import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import os
import glob
import random

class CVC_ClinicDB_loader(Dataset):
    def __init__(self,iw=384,ih=288,augmentation=False,phase='train'):
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
    
    def __get_path2__(self,phase,split=0.8):
        if not os.path.exists('./dataset/CVC-ClinicDB/'+phase+'.txt'):
            self.__gen_train_test(split=split)
        
    
    def __write_txt(self,fname,items):
        with open(fname, 'w') as fp:
            for item in items:
                # write each item on a new line
                fp.write("%s\n" % item)

    def __gen_train_test(self,split=0.8):
        list_dir = os.listdir('./dataset/CVC-ClinicDB/images/')
        img_path = './dataset/CVC-ClinicDB/images'
        mask_path = './dataset/CVC-ClinicDB/masks'
        all_img_path = [os.path.join(img_path,x) for x in list_dir]
        all_mask_path = [os.path.join(mask_path,x) for x in list_dir]
        indices = [x for x in range(len(all_img_path))]
        random.shuffle(indices)
        train_img = all_img_path[:int((len(indices))*split)]
        train_mask = all_mask_path[:int((len(indices))*split)]
        test_img = all_img_path[int((len(indices))*split):]
        test_mask = all_mask_path[int((len(indices))*split):]
        self.__write_txt(fname='./dataset/CVC-ClinicDB/train_img.txt',items=train_img)
        self.__write_txt(fname='./dataset/CVC-ClinicDB/train_mask.txt',items=train_mask)
        self.__write_txt(fname='./dataset/CVC-ClinicDB/test_img.txt',items=test_img)
        self.__write_txt(fname='./dataset/CVC-ClinicDB/test_mask.txt',items=test_mask)
    
    def __read_txt(self,path):
        with open(path) as f:
            list = [line.strip() for line in f.readlines()]
        return list
    
    def __get_path__(self,phase,split=0.8):
        if not os.path.exists('./dataset/CVC-ClinicDB/'+phase+'.txt'):
            self.__gen_train_test(split=split)
        if phase=='train':
            self.img = self.__read_txt('./dataset/CVC-ClinicDB/train_img.txt') 
            self.mask = self.__read_txt('./dataset/CVC-ClinicDB/train_mask.txt')          
        else:
            self.img = self.__read_txt('./dataset/CVC-ClinicDB/test_img.txt') 
            self.mask = self.__read_txt('./dataset/CVC-ClinicDB/test_mask.txt')     
    
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
        #init = np.expand_dims(init,axis=2)
        #init_mask = np.concatenate([~init,init],axis=2)
        #print(init_mask.shape)
        init_mask = self.mask_normalization(init_mask)
        init_mask = torch.cat([1-init_mask,init_mask],dim=0)
        img = self.img_normalization(img)
        label = self.mask_normalization(label)        
        
        return self.img[index], img, label, init_mask