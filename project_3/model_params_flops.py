from models import *
import argparse
from thop import profile
import torch


def model_params_flops(arch):
    print('==========================================================================')
    model = globals()[arch]()
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input,))
    print('==========================================================================')
    print('Total params:: {:.3f} M\n'
          'Total FLOPs: {:.3f}MFLOPs'.format(params/10**6, macs/10**6))
    print('==========================================================================')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the parameters and FLOPs of model')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vovnet27_slim',
                        help='model architecture')
    global args
    args = parser.parse_args()
    model_params_flops(args.arch)

