from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset

with open('./data/train_data', 'rb') as fo:
    dict = pickle.load(fo)
d = dict
data = d['data']
targets = d['labels']

data = np.dstack((data[:, :1024], data[:, 1024:2048], data[:, 2048:]))
data = data.reshape((data.shape[0], 32, 32, 3))

# count class
count_num = [0]*100
for i in range(len(targets)):
    count_num[targets[i]]+=1
print(count_num)


red = data[:, :, :, 0]
green = data[:, :, :, 1]
blue = data[:, :, :, 2]

print("means: [{}, {}, {}]".format(np.mean(red)/255.0, np.mean(green)/255.0, np.mean(blue)/255.0))
print("stdevs: [{}, {}, {}]".format(np.std(red)/255.0, np.std(green)/255.0, np.std(blue)/255.0))