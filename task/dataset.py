# -*- coding: utf-8 -*-
# @Author: Frank
# @Date:   2022-10-22 08:38:41
# @Last Modified by:   Frank
# @Last Modified time: 2024-03-15 18:40:30
import torch
import numpy as np 
from PIL import Image
from task import create_transform,create_transform1
import yacs.config
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GMM
from scipy import linalg
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms






def create_dataset(config):
    traindir1 = config.dataset.traindir1
    traindir2 = config.dataset.traindir2
    valdir = config.dataset.valdir
    train_transform,val_transform = create_transform1(config)

    train_dataset1 = RAFDBDataset(traindir1,
                                  transform = train_transform)
    train_dataset2 = AffectnetDataset(traindir2,
                                      transform = train_transform)
    val_dataset = RAFDBDataset_val(valdir,
                                   transform = val_transform)
    return train_dataset1, train_dataset2, val_dataset


class AffectnetDataset(torch.utils.data.Dataset):
    def __init__ (self, listPath,transform = None):
        self.transform = transform
        with open(listPath, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))


    def __getitem__(self, index):
        img_path, label1, valence, arousal = self.imgs[index] 
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label2 = float(valence)
        label3 = float(arousal)
        label2 = torch.from_numpy(np.array(float(label2))).float()
        label3 = torch.from_numpy(np.array(float(label3))).float()
        label2.reshape(1, )
        label3.reshape(1, )
        return img, torch.from_numpy(np.array(int(label1))), label2, label3

    def  __len__(self):
        return len(self.imgs)
    



class RAFDBDataset(torch.utils.data.Dataset):
    def __init__ (self, listPath,transform = None):
        self.transform = transform
        with open(listPath, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))

    def __getitem__(self, index):
        img_path, label1, valence, arousal = self.imgs[index] 
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label2 = float(valence)
        label3 = float(arousal)
        label2 = torch.from_numpy(np.array(float(label2))).float()
        label3 = torch.from_numpy(np.array(float(label3))).float()
        label2.reshape(1, )
        label3.reshape(1, )
        return img, torch.from_numpy(np.array(int(label1))), label2, label3

    def  __len__(self):
        return len(self.imgs)


    def getMiuCov(self):
        imgs = np.array(self.imgs)
        miu1 = imgs[:,2]
        miu1 = miu1.astype(float)
        miu2 = imgs[:,3].astype(float)
        miu = (np.mean(miu1),np.mean(miu2))
        cov = np.cov(miu1,miu2)
        return miu,cov 
        
    def getGaussianDict(self):
        emo_dict = {}
        imgs = np.array(self.imgs)
        for i in range(7):
            emo_dict['valence_{}'.format(i)] = []
            emo_dict['arousal_{}'.format(i)] = []
        for i in range(len(imgs)):
            emo_dict['valence_{}'.format(imgs[i][1])].append(float(imgs[i][2]))
            emo_dict['arousal_{}'.format(imgs[i][1])].append(float(imgs[i][3]))
        gaussian_dict = {}
        for i in range(7):
            gaussian_dict['miu_{}'.format(i)] = (np.mean(emo_dict['valence_{}'.format(i)]),np.mean(emo_dict['arousal_{}'.format(i)]))
            gaussian_dict['cov_{}'.format(i)] = np.cov(emo_dict['valence_{}'.format(i)], emo_dict['arousal_{}'.format(i)])    
        return gaussian_dict
        
        
        
class RAFDBDataset_val(torch.utils.data.Dataset):
    def __init__ (self, listPath,transform = None):
        self.transform = transform
        with open(listPath, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
    def __getitem__(self, index):
        img_path, label1= self.imgs[index] 
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, torch.from_numpy(np.array(int(label1)))

    def  __len__(self):
        return len(self.imgs)


