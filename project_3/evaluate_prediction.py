import argparse
import os
import time
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models import *
from imagenet100_32X32 import *
from functions import *
import pandas as pd
from image_preprocess import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet100_32*32 Evaluating')
parser.add_argument('--data', default='./data', type=str, metavar='N',
                    help='root directory of dataset where directory train_data or val_data exists')
parser.add_argument('--result', default='./Results/vov_arch_VoVNet_lr_0.2-06-10-23-06',
                    type=str, metavar='N', help='root directory of results')
parser.add_argument('--arch', '-a', metavar='ARCH', default='VoVNet',
                    help='model architecture')
# AlexNet_BN dutcvcnet
parser.add_argument('--num-classes', default=100, type=int,help='define the number of classes')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128) used for test')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model-dir', default='./Results/vov_arch_VoVNet_lr_0.2-06-10-23-06/model_best.pth.tar',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use.')


def main():
    global args
    args = parser.parse_args()
    # mkdir a new folder to store the checkpoint and best model
    if not os.path.exists(args.result):
        os.makedirs(args.result)


    # Model building
    print('=> Building model...')
    modeltype = globals()[args.arch]
    model = modeltype(num_classes=args.num_classes)
    # print(model)

    # optionally resume from a checkpoint
    if args.model_dir:
        if os.path.isfile(args.model_dir):
            print('=> loading checkpoint "{}"'.format(args.model_dir))
            if args.cuda:
                checkpoint = torch.load(args.model_dir)
            else:
                checkpoint = torch.load(args.model_dir, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.model_dir, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_dir))

    if args.cuda:
        print('GPU mode! ')
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    else:
        print('CPU mode! Cuda is not available!')

    # Data loading and preprocessing
    print('=> loading imagenet100 data...')

    test_dataset = ImageNet100_Test(
        root=args.data,
        transform=test_transforms())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2)
    # get prediction
    score = validate(test_loader, model)
    if args.cuda:
        score = score.cpu()

    # compute top1 accuracy
    _, pred = score.topk(1, 1, True, True)
    pred = pred.squeeze().numpy()

    # write scores to a csv file
    print('Writing scores to test_prediction.csv.......')
    csv_file = os.path.join(args.result, 'test_prediction.csv')
    imgs = []
    for i in range(len(test_dataset)):
        imgs.append('%d.png'%i)
    dataframe =pd.DataFrame({'Id': imgs, 'Prediction': pred})
    dataframe.to_csv(csv_file, index=False)
    print('Done!')


def validate(val_loader, model):
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    score = None
    ##################################################################################################################################
    split_num = 8
    score_temp = [0]*split_num
    max_temp = [0] * split_num
    with torch.no_grad():
        # end = time.time()
        for j in range(split_num):
            for i, (input, index) in enumerate(val_loader):
                if args.cuda:
                    input = input.cuda(non_blocking=True)
                if len(input.size()) > 4:  # 5-D tensor
                    bs, crops, ch, h, w = input.size()
                    output = model(input.view(-1, ch, h, w))
                    # fuse scores among all crops
                    output = output.view(bs, crops, -1).mean(dim=1)
                else:
                    output = model(input)
                if i == 0:
                    score_temp[j] = output
                else:
                    score_temp[j] = torch.cat([score_temp[j], output], dim=0)

                # measure elapsed time
                print(score_temp[j].shape)
                # print(score)
                # batch_time.update(time.time() - end)
                # end = time.time()
                # if i % args.print_freq == 0:
                #     print('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                #         i, len(val_loader), batch_time=batch_time))


            max_temp[j]= torch.max(score_temp[j],1)[0]

        cat_max_temp = None
        for k in range(split_num):
            if k ==0:
                cat_max_temp = torch.reshape( max_temp[k], (1, 30000))
            else:
                cat_max_temp = torch.cat([cat_max_temp, torch.reshape( max_temp[k], (1, 30000))], dim=0)




        # for i in range(30000):
        #     for j in range(8):

        index = torch.max(cat_max_temp, 0)[1]
        for l in range(30000):
            if l == 0:
                score = torch.reshape( score_temp[index[l]][l], (1, 100))
            else:
                score = torch.cat([score, torch.reshape( score_temp[index[l]][l], (1, 100))], dim=0)




    print(score.shape)
    return score

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

