# coding: utf-8
from PIL import Image
from numpy import *
from pylab import *
import numpy as np
import cv2


name = 'lab'
im1 = array(Image.open('./data/{:s}1.jpg'.format(name)))
im2 = array(Image.open('./data/{:s}2.jpg'.format(name)))
matches = np.loadtxt('./data/{:s}_matches.txt'.format(name))


x1 = matches[:, 0].T  # 将点集转化为齐次坐标表示
x2 = matches[:, 2].T
y1 = matches[:, 1].T  # 将点集转化为齐次坐标表示
y2 = matches[:, 3].T

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)

