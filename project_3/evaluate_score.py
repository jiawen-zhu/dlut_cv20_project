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
parser.add_argument('--arch', '-a', metavar='ARCH', default='AlexNet_BN',
                    help='model architecture: AlexNet_BN')
parser.add_argument('--num-classes', default=100, type=int, help='define the number of classes')
parser.add_argument('--result', default='',
                    type=str, metavar='N', help='root directory of results')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128) used for test')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model-dir', default='',
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
    print(model)

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
        score = score.cpu().numpy()
    else:
        score = score.numpy()

    # write scores to a csv file
    print('Wrinting scores to test_score.csv.......')
    csv_file = os.path.join(args.result, 'test_score.csv')
    imgs = []
    for i in range(len(test_dataset)):
        imgs.append('%d.png'%i)
    dataframe =pd.DataFrame(score, index=imgs,
                            columns=['score_%d'%i for i in range(args.num_classes)])
    dataframe.to_csv(csv_file)
    print('Done!')

def validate(val_loader, model):
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    score = None
    with torch.no_grad():
        end = time.time()
        for i, (input, index) in enumerate(val_loader):
            if args.cuda:
                input = input.cuda(non_blocking=True)

            # compute output
            if len(input.size()) > 4:  # 5-D tensor
                bs, crops, ch, h, w = input.size()
                output = model(input.view(-1, ch, h, w))
                # fuse scores among all crops
                output = output.view(bs, crops, -1).mean(dim=1)
            else:
                output = model(input)
            if i == 0:
                score = output
            else:
                score = torch.cat([score, output], dim=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time))

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

