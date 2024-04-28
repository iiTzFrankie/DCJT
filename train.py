# -*- coding: utf-8 -*-
# @Author: Frank
# @Date:   2022-10-22 17:09:15
# @Last Modified by:   Frank
# @Last Modified time: 2024-03-15 18:48:19
import os    
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import time
import datetime
import numpy as np
import pathlib
import random
import torch.backends.cudnn as cudnn

from task import (
    create_dataloader,
    create_logger,
    create_loss1,
    create_loss2,
    create_loss3,
    create_optimizer,
    create_scheduler,
    create_model
)

from task.utils import (
    AverageMeter,
    load_config,
    save_config,
    save_checkpoint,
    accuracy,
    ProgressMeter,
    RecorderMeter,

)




now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")



best_acc = 0
def main():
    global best_acc
    config = load_config()

    if not os.path.exists(os.path.dirname(config.train.output_dir)):
        os.makedirs(os.path.dirname(config.train.output_dir))
    
    if not os.path.exists(os.path.dirname(config.train.checkpoint_path)):
        os.makedirs(os.path.dirname(config.train.checkpoint_path))
        
    if config.train.seed != -1:
        random.seed(config.train.seed)
        torch.manual_seed(config.train.seed)
        torch.cuda.manual_seed_all(config.train.seed)
        np.random.seed(config.train.seed)
        cudnn.deterministic = True    

    output_dir = pathlib.Path(config.train.output_dir )
    save_config(config, output_dir,time_str)

    
    logger = create_logger(name=__name__,
                           output_dir = output_dir,
                           filename = time_str + 'log.txt')
    logger.info(config)
    train_loader1, train_loader2, val_loader = create_dataloader(config)
    model = create_model(config)

    criterion_cls, criterion_reg, criterion_mse = create_loss1(config),create_loss2(config),create_loss3(config)
    optimizer = create_optimizer(config,model)
    scheduler = create_scheduler(config,optimizer)
    model = torch.nn.DataParallel(model).cuda()
    





    for epoch in range(config.scheduler.start_epoch, config.scheduler.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'Current learning rate: {current_learning_rate}')
        train_acc1, train_los1,rmse1,rmse2 = train1(train_loader1, model, criterion_cls, criterion_reg, criterion_mse, optimizer, epoch, logger,config)
        
        
        
        if (epoch+1)%config.train.joint_frequency == 0 and epoch > 29 :
            train_los2,rmse1,rmse2 = train2(train_loader2, model, criterion_cls,criterion_reg, criterion_mse, optimizer, epoch, logger, config)

        
        val_acc, val_los = validate(val_loader, model, criterion_cls, criterion_reg, criterion_mse, logger, config)
        scheduler.step()
        

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        logger.info(f'Current best accuracy: {best_acc.item()}')
        
    
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         }, is_best, time_str,config)
        end_time = time.time()
        epoch_time = end_time - start_time
    
        


def train1(train_loader, model, criterion_cls, criterion_reg, criterion_mse, optimizer, epoch, logger,config):
   
    losses = AverageMeter('Loss', ':.4f')
    mseMeter_valence = AverageMeter('mseValence', ':.4f')
    mseMeter_arousal = AverageMeter('mseArousal', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1, mseMeter_valence, mseMeter_arousal],logger,
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target_cls,valence,arousal) in enumerate(train_loader):
        num = images.size(0)
        images = images.cuda()
        target_cls = target_cls.cuda()
        valence = valence.cuda().reshape(num, -1)
        arousal = arousal.cuda().reshape(num, -1)

        
        target_reg = torch.cat((valence,arousal),1)
        # compute output
        output = model(images)
        output_cls, output_reg = output[:, 0:7], output[:, 7:9]
        #loss = sigma1 + sigma2 + \
               #torch.exp(-sigma2) * (args.beta * criterion_cls(output1_cls, target_cls)) + ((1-args.beta) * criterion_cls(output2_cls, target_cls)) + \
               #torch.exp(-sigma1) * (args.beta * criterion_reg(output1_reg, target_reg)) + ((1-args.beta) * criterion_reg(output2_reg, target_reg))
        loss = criterion_cls(output_cls, target_cls) + config.model.reg_weight * criterion_reg(output_reg, target_reg) 
        MSE_valence = criterion_mse(output_reg[:,0:1],target_reg[:,0:1])
        MSE_arousal = criterion_mse(output_reg[:,1:2],target_reg[:,1:2])
        # measure accuracy and record loss
        acc1, _ = accuracy(output_cls[:,0:7], target_cls, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        mseMeter_valence.update(MSE_valence.item(), num)
        mseMeter_arousal.update(MSE_arousal.item(), num)

        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % config.train.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg, np.sqrt(mseMeter_valence.avg), np.sqrt(mseMeter_arousal.avg)



def train2(train_loader, model, criterion_cls, criterion_reg, criterion_mse, optimizer, epoch, logger,config):
    losses = AverageMeter('Loss', ':.4f')
    mseMeter_valence = AverageMeter('mseValence', ':.4f')
    mseMeter_arousal = AverageMeter('mseArousal', ':.4f')
    progress = ProgressMeter(len(train_loader),
                             [losses, mseMeter_valence, mseMeter_arousal],logger,
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target_cls,valence,arousal) in enumerate(train_loader):
        num = images.size(0)
        images = images.cuda()
        target_cls = target_cls.cuda()
        valence = valence.cuda().reshape(num, -1)
        arousal = arousal.cuda().reshape(num, -1)
        
        target_reg = torch.cat((valence,arousal),1)
        # compute output
        output = model(images)
        output_cls = output[:, 0:7]
        output_reg = output[:, 7:9]

        loss = criterion_cls(output_cls, target_cls) + config.model.reg_weight * criterion_reg(output_reg, target_reg) 
        MSE_valence = criterion_mse(output_reg[:,0:1],target_reg[:,0:1])
        MSE_arousal = criterion_mse(output_reg[:,1:2],target_reg[:,1:2])
        # measure accuracy and record loss

        losses.update(loss.item(), images.size(0))
        mseMeter_valence.update(MSE_valence.item(), num)
        mseMeter_arousal.update(MSE_arousal.item(), num)

        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % 200 == 0:
            progress.display(i)

    return losses.avg, np.sqrt(mseMeter_valence.avg), np.sqrt(mseMeter_arousal.avg)









def validate(val_loader, model, criterion_cls, criterion_reg, criterion_mse, logger,config):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],logger,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target_cls) in enumerate(val_loader):
            num = images.size(0)
            images = images.cuda()
            target_cls = target_cls.cuda()

            
          
            # compute output                        
            output = model(images)
            output_cls, output_reg = output[:, 0:7], output[:, 7:9]
            loss = criterion_cls(output_cls, target_cls)
               

            
            
            # measure accuracy and record loss
            acc, _ = accuracy(output[:,0:7], target_cls, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            
            if i % config.train.print_freq == 0:
                progress.display(i)
        logger.info(f' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        #print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        #with open('./log/'+args.experiment_mode  + time_str + 'log.txt', 'a') as f:
            #f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n' )
    return top1.avg, losses.avg




if __name__ == '__main__':
    main()

















