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

parser = argparse.ArgumentParser(description='PyTorch ImageNet100_32*32 Training')
parser.add_argument('--data', default='./data', type=str, metavar='N',
                    help='root directory of dataset where directory train_data or val_data exists')
parser.add_argument('--result', default='./Results',
                    type=str, metavar='N', help='root directory of results')
#*********************************************************************
parser.add_argument('--arch', '-a', metavar='ARCH', default='dutcvcnet',
                    help='model architecture')
# dutcvcnet VoVNet
# AlexNet_BN PeleeNet HBONet vovnet27_slim ghost_net MobileNetV3_Large VoVNet
parser.add_argument('--num-classes', default=100, type=int, help='define the number of classes')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')#140
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128), used for train and validation')
# 128
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float, metavar='LR',
                    help='initial learning rate') # 0.1
# alex 0.1 pelee 0.18 pelee_adam 1e-4 vovnet27_slim 0.1(SGD) 1e-3(adam)
parser.add_argument('--optimizer', default='SGD', type=str, metavar='M', help='optimization method')
# SGD0.1 Adam1e-3 RMSprop0.01
parser.add_argument('--print-freq', '-p', default=80, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--save-freq', '-sp', default=40, type=int, metavar='N',
                    help='save checkpoint frequency (default: 10)')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default=True, action='store_true', help='pretrained')
# ./Results/AlexNet_BN_lr_0.1/checkpoint_10.pth.tar
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use.')
parser.add_argument('--adjust_lr', default='step_decrease', type=str, help='way to adjust lr')
# step_decrease cosine warm_up
#****************************************************************************
parser.add_argument('--label_smooth', default=False, action='store_true', help='label_smooth')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser.add_argument('--task', default='step_decrease', type=str, help='task')
# parser.add_argument('--model_type', default='vovnet19_1', type=str, help='model_type')
# parser.add_argument('--head', default=0, type=int, help='head')
# parser.add_argument('--device', default=0, type=int, help='device')
# vov_arch
best_prec1 = 0

np.random.seed(3035)
torch.manual_seed(3035)
torch.cuda.manual_seed(3035)

# torch.backends.cudnn.enabled = True


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.start_epoch = 0
    global device
    # device = torch.device('cuda:' + str(args.device))
    # mkdir a new folder to store the checkpoint and best model

    args.result = os.path.join(args.result, args.task + '_' + args.arch + '_lr_{}'.format(args.lr)
                               + time.strftime("-%m-%d-%H-%M", time.localtime()))
    print(args)
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # Model building
    print('=> Building model...')
    modeltype = globals()[args.arch]
    # model = modeltype(num_classes=args.num_classes)
    ######################################################################################
    # model = modeltype(num_classes=args.num_classes, head=args.head, model_type=args.model_type)
    model = modeltype(num_classes=args.num_classes)
    # print(model)


    # compute the parameters and FLOPs of model
    model_params_flops(args.arch,args)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    # criterion = None
    criterion2 = None

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

    if args.pretrained:
        pretrained_path0 = './pretrained_model/vovnet19-72.03-top1.pt'
        if os.path.isfile(pretrained_path0):
            print('=> pretrained_model ')
            if args.cuda:
                pretrained_model = torch.load(pretrained_path0)
            else:
                pretrained_model = torch.load(pretrained_path0, map_location=torch.device('cpu'))
            model_dict = model.state_dict()

            # for k, v in pretrained_model.items():
            #     if k in model_dict and model_dict[k].size()==v.size():
            #         print(k)

            pretrained_dict = {k: v for k, v in pretrained_model.items() if
                               k in model_dict and model_dict[k].size()==v.size()}  # filter out unnecessary keys
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            print("=> no pretrained_model found at ")

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), args.lr, alpha=0.9)
    else:
        raise KeyError('optimization method {} is not achieved')


    if args.cuda:
        print('GPU mode! ')
        ##########################################
        # model = model.cuda()
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        ########################################
        model = nn.DataParallel(model).cuda()
        # model = nn.DataParallel(model).to(device)

        if args.label_smooth:
            criterion = LabelSmoothing(smoothing=0.15).cuda()
            # criterion = LabelSmoothing(smoothing=0.15).to(device)
            # criterion2 = criterion.cuda()
            print('label smooth!')
        else:
            criterion = criterion.cuda()
            # criterion = criterion.to(device)
        cudnn.benchmark = True
    else:
        print('CPU mode! Cuda is not available!')



    #############################



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
        adjust_learning_rate(optimizer, epoch, iterations_per_epoch=len(train_loader),
                             iteration=epoch * len(train_loader))
        print('learning rate:{}'.format(optimizer.param_groups[0]['lr']))
        # train for one epoch
        tic = time.time()
        #################################################################
        trainObj, top1, top5 = train(train_loader, model, criterion, optimizer, epoch, criterion2=criterion2)
        print("###########cost time:", time.time() - tic)
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
        # if int(epoch + 1) % args.save_freq == 0:
        #     print("=> save checkpoint_{}.pth.tar'".format(int(epoch + 1)))
        #     save_checkpoint(stat, False,
        #                     [os.path.join(args.result, 'checkpoint_{}.pth.tar'.format(int(epoch + 1)))])
        # plot curve
        plot_curve(stats_, args.result, False)
        data = stats_
        sio.savemat(os.path.join(args.result, 'stats.mat'), {'data': data})


def train(train_loader, model, criterion, optimizer, epoch, criterion2=None):
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

        if args.cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # input = input.to(device)
            # target = target.to(device)

        # compute output
        output = model(input)
        # print(target.shape)
        loss = criterion(output, target)
        #####################################################################
        # loss = 0.5*criterion(output, target) + 0.5*criterion2(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        ######################################################
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
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
                # input = input.to(device)
                # target = target.to(device)


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
    # torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])


def adjust_learning_rate(optimizer, epoch, iterations_per_epoch=None, iteration=None):
    """
     For AlexNet, the lr starts from 0.05, and is divided by 10 at 90 and 120 epochs
    """
    if args.adjust_lr == 'step_decrease':
        print(args.adjust_lr + ' learn rate policy ')
        if epoch < 100:  # 90
            lr = args.lr
        elif epoch < 110:
            lr = args.lr * 0.1
        elif epoch < 115:
            lr = args.lr * 0.01
        else:
            lr = args.lr * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.adjust_lr == 'cosine':
        print(args.adjust_lr + ' learn rate policy ')
        T_total = args.epochs * iterations_per_epoch
        T_cur = iteration # (epoch % args.epochs) * iterations_per_epoch
        # print('T_cur / T_total:  ', T_cur / T_total)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.adjust_lr == 'warm_up':
        print(args.adjust_lr + ' learn rate policy ')
        if epoch < 3:
            lr = args.lr * 0.001 * (10**epoch)
        elif epoch < 100:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        elif epoch < 135:
            lr = args.lr * 0.01
        else:
            lr = args.lr * 0.001
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


if __name__ == '__main__':
    main()
