import os
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np



class stats:
    def __init__(self, path, start_epoch):
        if start_epoch is not 0:
           stats_ = sio.loadmat(os.path.join(path,'stats.mat'))
           data = stats_['data']
           content = data[0,0]
           self.trainObj = content['trainObj'][:,:start_epoch].squeeze().tolist()
           self.trainTop1 = content['trainTop1'][:,:start_epoch].squeeze().tolist()
           self.trainTop5 = content['trainTop5'][:,:start_epoch].squeeze().tolist()
           self.valObj = content['valObj'][:,:start_epoch].squeeze().tolist()
           self.valTop1 = content['valTop1'][:,:start_epoch].squeeze().tolist()
           self.valTop5 = content['valTop5'][:,:start_epoch].squeeze().tolist()
           if start_epoch is 1:
               self.trainObj = [self.trainObj]
               self.trainTop1 = [self.trainTop1]
               self.trainTop5 = [self.trainTop5]
               self.valObj = [self.valObj]
               self.valTop1 = [self.valTop1]
               self.valTop5 = [self.valTop5]
        else:
           self.trainObj = []
           self.trainTop1 = []
           self.trainTop5 = []
           self.valObj = []
           self.valTop1 = []
           self.valTop5 = []
    def _update(self, trainObj, top1, top5, valObj, prec1, prec5):
        self.trainObj.append(trainObj)
        self.trainTop1.append(top1.cpu().numpy())
        self.trainTop5.append(top5.cpu().numpy())
        self.valObj.append(valObj)
        self.valTop1.append(prec1.cpu().numpy())
        self.valTop5.append(prec5.cpu().numpy())



def plot_curve(stats, path, iserr):
    trainObj = np.array(stats.trainObj)
    valObj = np.array(stats.valObj)
    if iserr:
        trainTop1 = 100 - np.array(stats.trainTop1)
        trainTop5 = 100 - np.array(stats.trainTop5)
        valTop1 = 100 - np.array(stats.valTop1)
        valTop5 = 100 - np.array(stats.valTop5)
        titleName = 'error'
    else:
        trainTop1 = np.array(stats.trainTop1)
        trainTop5 = np.array(stats.trainTop5)
        valTop1 = np.array(stats.valTop1)
        valTop5 = np.array(stats.valTop5)
        titleName = 'accuracy'
        # best_top1 = 0
    epoch = len(trainObj)
    figure = plt.figure()
    obj = plt.subplot(1,3,1)
    obj.plot(range(1,epoch+1),trainObj,'o-',label = 'train')
    obj.plot(range(1,epoch+1),valObj,'o-',label = 'val')
    plt.xlabel('epoch')
    plt.title('objective')
    handles, labels = obj.get_legend_handles_labels()
    obj.legend(handles[::-1], labels[::-1])
    top1 = plt.subplot(1,3,2)
    top1.plot(range(1,epoch+1),trainTop1,'o-',label = 'train')
    top1.plot(range(1,epoch+1),valTop1,'o-',label = 'val')
    plt.title('top1'+titleName+ ' best:' + str(round(max(valTop1),2)))
    plt.xlabel('epoch')
    # top1.text(0, 0, 'best=53.01', fontsize=15)
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])
    top5 = plt.subplot(1,3,3)
    top5.plot(range(1,epoch+1),trainTop5,'o-',label = 'train')
    top5.plot(range(1,epoch+1),valTop5,'o-',label = 'val')
    plt.title('top5'+titleName)
    plt.xlabel('epoch')
    handles, labels = top5.get_legend_handles_labels()
    top5.legend(handles[::-1], labels[::-1])
    filename = os.path.join(path, 'net-train.pdf')
    figure.savefig(filename, bbox_inches='tight')
    plt.close()

def decode_params(input_params):
    params = input_params[0]
    out_params = []
    _start=0
    _end=0
    for i in range(len(params)):
        if params[i] == ',':
            out_params.append(float(params[_start:_end]))
            _start=_end+1
        _end+=1
    out_params.append(float(params[_start:_end]))
    return out_params
