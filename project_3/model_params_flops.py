from models import *
import argparse
from thop import profile
import torch


def model_params_flops(arch,args):
    print('==========================================================================')
    model = globals()[arch](head=args.head, model_type=args.model_type)

    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input,))
    print(model)
    print('==========================================================================')
    print('Total params:: {:.3f} M\n'
          'Total FLOPs: {:.3f}MFLOPs'.format(params/10**6, macs/10**6))
    print('==========================================================================')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the parameters and FLOPs of model')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='VoVNet',
                        help='model architecture')
    parser.add_argument('--model_type', default='vovnet19_1', type=str, help='model_type')
    parser.add_argument('--head', default=0, type=int, help='head')


    global args
    args = parser.parse_args()
    model_params_flops(args.arch, args)

