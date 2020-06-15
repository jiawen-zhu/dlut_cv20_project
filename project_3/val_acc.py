import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
# from apex import amp

from models import *
from imagenet100_32X32 import *
from functions import *
from label_smooth import *
from image_preprocess import *
from model_params_flops import *

def main():
    modeltype = globals()['dutcvcnet']
    model = modeltype(num_classes=100)
    model_dir = './Results/dlutcvc_log1/model_best.pth.tar'
    if model_dir:
        if os.path.isfile(model_dir):
            print('=> loading checkpoint ')
            checkpoint = torch.load(model_dir)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(model_dir, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_dir))
    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # _, val_transforms = train_val_transforms()
    # val_transforms = test_transforms()
    # val_dataset = ImageNet100(
    #     root='./data',
    #     train=False,
    #     transform=val_transforms)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128,
    #                                           shuffle=False, num_workers=2)

    top1, top5 = validate(model)



#####################################




def validate(model):
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    split_num = 16 #至少16,前期j=0有效,16以后就没有了
    score_temp = [0]*split_num
    max_temp = [0] * split_num
    target_all = None
    with torch.no_grad():
        # end = time.time()
        for j in range(split_num):
            # if j == 0:
            #     _, val_transforms = train_val_transforms()
            # else:
            val_transforms = test_transforms()
            val_dataset = ImageNet100(
                root='./data',
                train=False,
                transform=val_transforms)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128,
                                                      shuffle=False, num_workers=2)

            for i, (input, target) in enumerate(val_loader):

                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                if j==0:
                    if i==0:
                        target_all = target
                    else:
                        target_all = torch.cat([target_all, target], dim=0)

                # compute output
                output = model(input)
                # print(output.size())
                # print(target.size())
                if i == 0:
                    score_temp[j] = output
                else:
                    score_temp[j] = torch.cat([score_temp[j], output], dim=0)

            print(target_all.size())
            max_temp[j] = torch.max(score_temp[j], 1)[0]


        ###########################################################mean********
        cat_score_temp = None
        for k in range(split_num):
            if k ==0:
                cat_score_temp = torch.reshape(score_temp[k], (1, 10000, 100))
            else:
                cat_score_temp = torch.cat([cat_score_temp, torch.reshape(score_temp[k], (1, 10000, 100))], dim=0)
        score = cat_score_temp.mean(0)
        #######################################################

########################################################################max
        # cat_max_temp = None
        # for k in range(split_num):
        #     if k ==0:
        #         cat_max_temp = torch.reshape( max_temp[k], (1, 10000))
        #     else:
        #         cat_max_temp = torch.cat([cat_max_temp, torch.reshape( max_temp[k], (1, 10000))], dim=0)
        #
        # index = torch.max(cat_max_temp, 0)[1]
        # for l in range(10000):
        #     if l == 0:
        #         score = torch.reshape( score_temp[index[l]][l], (1, 100))
        #     else:
        #         score = torch.cat([score, torch.reshape( score_temp[index[l]][l], (1, 100))], dim=0)
###################################################################################


        # measure accuracy and record loss
        prec1, prec5 = accuracy(score, target_all, topk=(1, 5))
        top1.update(prec1[0], score.size(0))
        top5.update(prec5[0], score.size(0))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__=='__main__':
    main()