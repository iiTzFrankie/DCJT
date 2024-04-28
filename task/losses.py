import torch.nn as nn
import yacs.config
from .typess import LossType

def create_loss1(config):
    loss_name = config.train.loss1
    if loss_name == LossType.L1.name:
        return nn.L1Loss(reduction='mean')
    elif loss_name == LossType.L2.name:
        return nn.MSELoss(reduction='mean')
    elif loss_name == LossType.SmoothL1.name:
        return nn.SmoothL1Loss(reduction='mean')
    elif loss_name == LossType.CrossEntropy.name:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError    

def create_loss2(config):
    loss_name = config.train.loss2
    if loss_name == LossType.L1.name:
        return nn.L1Loss(reduction='mean')
    elif loss_name == LossType.L2.name:
        return nn.MSELoss(reduction='mean')
    elif loss_name == LossType.SmoothL1.name:
        return nn.SmoothL1Loss(reduction='mean')
    elif loss_name == LossType.CrossEntropy.name:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError  
        

def create_loss3(config):
    loss_name = config.train.loss3
    if loss_name == LossType.L1.name:
        return nn.L1Loss(reduction='mean')
    elif loss_name == LossType.L2.name:
        return nn.MSELoss(reduction='mean')
    elif loss_name == LossType.SmoothL1.name:
        return nn.SmoothL1Loss(reduction='mean')
    elif loss_name == LossType.CrossEntropy.name:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError  