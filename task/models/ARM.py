# -*- coding: utf-8 -*-
# @Author: Frank
# @Date:   2023-05-11 15:43:33
# @Last Modified by:   Frank
# @Last Modified time: 2023-06-27 15:06:05
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=9, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, out



class ResNet18_ARM___RAF(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(ResNet18_ARM___RAF, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        #checkpoint = torch.load("/home/frank/VACFER_MMD/models/resnet18_msceleb.pth")
        #resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # before avgpool 512x1
        self.arrangement = nn.PixelShuffle(16)
        self.arm = Amend_raf()
        self.fc = nn.Linear(121, num_classes)


    def forward(self, x):
        x = self.features(x)
        print(x.shape)

        x = self.arrangement(x)

        x, alpha = self.arm(x)
        print(x.shape)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, alpha

class Amend_raf(nn.Module):  # moren
    def __init__(self, inplace=2):
        super(Amend_raf, self).__init__()
        self.de_albino = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=32, stride=8, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(inplace)
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        mask = torch.tensor([]).cuda()
        createVar = locals()
        for i in range(x.size(1)):
            createVar['x' + str(i)] = torch.unsqueeze(x[:, i], 1)
            createVar['x' + str(i)] = self.de_albino(createVar['x' + str(i)])
            mask = torch.cat((mask, createVar['x' + str(i)]), 1)
        x = self.bn(mask)
        xmax, _ = torch.max(x, 1, keepdim=True)
        global_mean = x.mean(dim=[0, 1])
        xmean = torch.mean(x, 1, keepdim=True)
        xmin, _ = torch.min(x, 1, keepdim=True)
        x = xmean + self.alpha * global_mean

        return x, self.alpha


def Model():
    model = ResNet18_ARM___RAF()
    model = torch.nn.DataParallel(model).cuda()
    #checkpoint = torch.load("/home/frank/subFolder/CC-FER202204/checkpoints/pretrain/[05-05]-[09-39]-model_best_epoch60.pth")
    #pre_trained_dict = checkpoint['state_dict']
    #model.load_state_dict(pre_trained_dict)
    return model

if __name__ == '__main__':
    model = ResNet18_ARM___RAF().cuda()
    X = torch.randn(1,3,224,224).cuda()
    output = model(X)
    print(output)