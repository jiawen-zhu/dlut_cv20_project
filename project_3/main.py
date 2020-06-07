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

from models import *
from imagenet100_32X32 import *
from functions import *
from image_preprocess import *
from model_params_flops import *


parser = argparse.ArgumentParser(description='PyTorch ImageNet100_32*32 Training')
parser.add_argument('--data', default='./data', type=str, metavar='N',
                    help='root directory of dataset where directory train_data or val_data exists')
parser.add_argument('--result', default='./Results',
                    type=str, metavar='N', help='root directory of results')
parser.add_argument('--arch', '-a', metavar='ARCH', default='AlexNet_BN',
                    help='model architecture: AlexNet_BN')
parser.add_argument('--num-classes', default=100, type=int, help='define the number of classes')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128), used for train and validation')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='M', help='optimization method')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--save-freq', '-sp', default=10, type=int, metavar='N',
                    help='save checkpoint frequency (default: 10)')
parser.add_argument('--resume', default='./Results/AlexNet_BN_lr_0.1/checkpoint_10.pth.tar',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use.')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.start_epoch = 0
    # mkdir a new folder to store the checkpoint and best model
    args.result = os.path.join(args.result, args.arch + '_lr_{}'.format(args.lr))
    print(args)
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # Model building
    print('=> Building model...')
    modeltype = globals()[args.arch]
    model = modeltype(num_classes=args.num_classes)
    print(model)

    # compute the parameters and FLOPs of model
    model_params_flops(args.arch)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            if args.cuda:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_load_state_dict = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        print('GPU mode! ')
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    else:
        print('CPU mode! Cuda is not available!')

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'custom':
        """
            You can achieve your own optimizer here
        """
        pass
    else:
        raise KeyError('optimization method {} is not achieved')

    if args.resume:
        if os.path.isfile(args.resume):
            optimizer.load_state_dict(optimizer_load_state_dict)

    # Data loading and preprocessing
    print('=> loading imagenet100 data...')
    train_transforms, val_transforms = train_val_transforms()
    train_dataset = ImageNet100(
        root=args.data,
        train=True,
        transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)

    val_dataset = ImageNet100(
        root=args.data,
        train=False,
        transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    stats_ = stats(args.result, args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('learning rate:{}'.format(optimizer.param_groups[0]['lr']))
        # train for one epoch
        trainObj, top1, top5 = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        valObj, prec1, prec5 = validate(val_loader, model, criterion)
        # update stats
        stats_._update(trainObj, top1, top5, valObj, prec1, prec5)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = []
        filename.append(os.path.join(args.result, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.result, 'model_best.pth.tar'))
        stat = {'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()}
        save_checkpoint(stat, is_best, filename)
        if int(epoch+1) % args.save_freq == 0:
            print("=> save checkpoint_{}.pth.tar'".format(int(epoch + 1)))
            save_checkpoint(stat, False,
                            [os.path.join(args.result, 'checkpoint_{}.pth.tar'.format(int(epoch + 1)))])
        #plot curve
        plot_curve(stats_, args.result, True)
        data = stats_
        sio.savemat(os.path.join(args.result, 'stats.mat'), {'data': data})



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda :
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top1.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    state_dict = state['state_dict']
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state = OrderedDict()
        for key in state_dict.keys():
            if 'module.' in key:
                new_key = key.replace('module.', '')
                new_state[new_key] = state_dict[key]
            else:
                new_state[key] = state_dict[key]
        state_dict = new_state
    state['state_dict'] = state_dict
    torch.save(state, filename[0])
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])


def adjust_learning_rate(optimizer, epoch):
    """
     For AlexNet, the lr starts from 0.05, and is divided by 10 at 90 and 120 epochs
    """
    if epoch < 90:
        lr = args.lr
    elif epoch < 120:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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


if __name__=='__main__':
    main()

