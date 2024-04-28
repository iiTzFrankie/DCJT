import torch
import yacs.config

def create_scheduler(config, optimizer):
    if config.scheduler.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.lr_decay)
    elif config.scheduler.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.epochs,
            eta_min=config.scheduler.lr_min_factor)
    elif config.scheduler.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.lr_decay)
    elif config.scheduler.type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler.lr_decay)
    else:
        raise ValueError
    return scheduler
