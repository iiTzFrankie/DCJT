# -*- coding: utf-8 -*-
# @Author: Frank
# @Date:   2023-05-11 15:43:33
# @Last Modified by:   Frank
# @Last Modified time: 2023-07-10 19:48:30
import torchvision
import torchvision.transforms as transforms
def create_transform(config):
    train_transforms = torchvision.transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        #transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
        #transforms.RandomErasing(scale=(0.02,0.25))
        ])
    
    val_transforms = torchvision.transforms.Compose([
                                   transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
                                                        ])
                               

    return train_transforms, val_transforms



def create_transform1(config):
    train_transforms = torchvision.transforms.Compose([
        #transforms.RandomResizedCrop((224, 224)),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))
        ])
    
    val_transforms = torchvision.transforms.Compose([
                                   transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
                                                        ])
                               

    return train_transforms, val_transforms